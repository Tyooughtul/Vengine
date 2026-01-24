#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <set>
#include "../src/core/dataset.hpp"
#include "../src/core/metrics.hpp"
#include "../src/core/ivf_index.hpp"

// 生成随机向量
std::vector<float> generate_random_vector(int dim) {
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> vec(dim);
    for(int i=0; i<dim; ++i) vec[i] = dist(rng);
    return vec;
}

int main() {
    const int DIM = 128;
    const int N_VECTORS = 20000; 
    const int N_QUERIES = 10;   
    const int K = 10;           // Top-10
    const int N_LISTS = 100;    // 聚类成 100 个桶
    const int N_PROBE = 10;      // 搜索时只看最近的 10 个桶 (10% 的数据)

    std::cout << "=== Mini-Milvus Benchmark ===" << std::endl;
    std::cout << "Dataset: " << N_VECTORS << " vectors, Dim=" << DIM << std::endl;

    // 1. Prepare Data
    std::cout << "[1] Generating data..." << std::endl;
    minimilvus::VectorDataset dataset(DIM);
    for(int i=0; i<N_VECTORS; ++i) {
        dataset.add(generate_random_vector(DIM));
    }
    
    // 生成一些查询向量
    std::vector<std::vector<float>> queries;
    for(int i=0; i<N_QUERIES; ++i) {
        queries.push_back(generate_random_vector(DIM));
    }

    // --- Brute Force Search ---
    std::cout << "[2] Running Brute Force Search (Baseline)..." << std::endl;
    std::vector<std::set<int64_t>> ground_truth; // 存下来用于算 Recall

    auto start_bf = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        // 简单的暴力搜索 TopK (用最小堆逻辑自己写一个简单的)
        std::priority_queue<minimilvus::SearchResult> pq;
        std::span<const float> q_span(q.data(), DIM);
        
        for(int i=0; i<N_VECTORS; ++i) {
            float d = minimilvus::l2_distance(q_span, dataset.get_vector(i));
            if(pq.size() < K) {
                pq.push({(int64_t)i, d});
            } else if (d < pq.top().distance) {
                pq.pop();
                pq.push({(int64_t)i, d});
            }
        }
        
        // 保存这一组的正确答案 ID
        std::set<int64_t> truth;
        while(!pq.empty()) {
            truth.insert(pq.top().id);
            pq.pop();
        }
        ground_truth.push_back(truth);
    }
    auto end_bf = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_bf = end_bf - start_bf;
    std::cout << "    -> Brute Force Time: " << time_bf.count() << "s" << std::endl;


    // --- IVF Search ---
    std::cout << "[3] Building IVF Index..." << std::endl;
    auto start_build = std::chrono::high_resolution_clock::now();
    
    minimilvus::IVFIndex index(DIM, N_LISTS);
    index.build(dataset);
    
    auto end_build = std::chrono::high_resolution_clock::now();
    std::cout << "    -> Build Time: " << std::chrono::duration<double>(end_build - start_build).count() << "s" << std::endl;

    std::cout << "[4] Running IVF Search..." << std::endl;
    float total_recall = 0;
    
    auto start_ivf = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_QUERIES; ++i) {
        std::span<const float> q_span(queries[i].data(), DIM);
        auto results = index.search(q_span, dataset, K, N_PROBE);
        
        // 计算 Recall (召回率)
        // 看搜出来的 ID 有多少在 Ground Truth 里
        int hit = 0;
        for (const auto& res : results) {
            if (ground_truth[i].count(res.id)) {
                hit++;
            }
        }
        total_recall += (float)hit / K;
    }
    auto end_ivf = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ivf = end_ivf - start_ivf;
    
    std::cout << "    -> IVF Search Time: " << time_ivf.count() << "s" << std::endl;
    std::cout << "    -> Speedup: " << time_bf.count() / time_ivf.count() << "x" << std::endl;
    std::cout << "    -> Avg Recall: " << (total_recall / N_QUERIES) * 100 << "%" << std::endl;

    return 0;
}