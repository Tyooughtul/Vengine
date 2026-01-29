#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <set>
#include "../src/core/dataset.hpp"
#include "../src/core/metrics.hpp"
#include "../src/core/ivf_index.hpp"

// --- 将原来的 generate_random_vector 替换/补充为 generate_clustered_data ---

struct DataGenerator {
    std::mt19937 rng{42};
    std::vector<std::vector<float>> centers;
    int dim;
    
    DataGenerator(int k_centers, int d) : dim(d) {
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for(int i=0; i<k_centers; ++i) {
            std::vector<float> c(dim);
            for(int j=0; j<dim; ++j) c[j] = dist(rng);
            centers.push_back(c);
        }
    }
    
    std::vector<float> generate() {
        // 随机选一个簇中心
        std::uniform_int_distribution<int> center_dist(0, centers.size()-1);
        const auto& center = centers[center_dist(rng)];
        
        // 在中心附近加高斯噪声
        std::normal_distribution<float> noise_dist(0.0f, 1.0f); // 标准差1.0
        std::vector<float> vec(dim);
        for(int i=0; i<dim; ++i) {
            vec[i] = center[i] + noise_dist(rng);
        }
        return vec;
    }
};

int main() {
    const int DIM = 128;
    const int N_VECTORS = 1000000; 
    const int N_QUERIES = 100;   
    const int K = 10;           
    const int N_LISTS = 1000;    
    
    // 搜索参数
    const float PROBE_RATIO = 0.2f; // 允许距离最近桶距离扩大20%
    const int MAX_PROBE = 20;       // 最多搜20个桶
    const int REFINE_FACTOR = 5;    // 精排因子

    std::cout << "=== Mini-Milvus Benchmark (Clustered Data) ===" << std::endl;
    
    // 使用高斯混合数据生成器 (10个中心)
    DataGenerator generator(100, DIM); 
    minimilvus::VectorDataset dataset(DIM);
    for(int i=0; i<N_VECTORS; ++i) {
        dataset.add(generator.generate());
    }
    
    std::vector<std::vector<float>> queries;
    for(int i=0; i<N_QUERIES; ++i) {
        queries.push_back(generator.generate());
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

    std::cout << "[4] Running Smart IVF Search..." << std::endl;
    float total_recall = 0;
    
    auto start_ivf = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_QUERIES; ++i) {
        std::span<const float> q_span(queries[i].data(), DIM);
        auto results = index.search(q_span, dataset, K, PROBE_RATIO, MAX_PROBE, REFINE_FACTOR);
        
        // 计算 Recall (召回率)
        // 看搜出来的 ID 有多少在 Ground Truth 里
        int hit = 0;
        for (const auto& res : results) {
            if (ground_truth[i].count(res.id)) {
                hit++;
            }
        }
        std::cout << "Single Time Recall: " << (float)hit / K << std::endl;
        total_recall += (float)hit / K;
    }
    auto end_ivf = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ivf = end_ivf - start_ivf;
    
    std::cout << "    -> IVF Search Time: " << time_ivf.count() << "s" << std::endl;
    std::cout << "    -> Speedup: " << time_bf.count() / time_ivf.count() << "x" << std::endl;
    std::cout << "    -> Avg Recall: " << (total_recall / N_QUERIES) * 100 << "%" << std::endl;

    return 0;
}