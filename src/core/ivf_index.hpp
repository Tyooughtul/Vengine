/**
 * @file    ivf_index.hpp
 * @brief   IVF索引实现
 * @details 基于KMeans聚类的向量索引，通过分桶大幅减少搜索空间
 * @author  Tyooughtul
 */

#pragma once
#include <vector>
#include <algorithm>
#include <queue>
#include <omp.h>
#include "kmeans.hpp"
#include "dataset.hpp"
#include "metrics.hpp"

namespace minimilvus {

/**
 * @brief   搜索结果结构
 * @details 存储单个搜索结果，包含向量ID和与查询向量的距离
 */
struct SearchResult {
    idx_t id;        ///< 向量ID
    float distance;  ///< 与查询向量的距离
    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }
};

/**
 * @brief   IVF索引类
 * @details 将向量分配到多个倒排桶中，搜索时只扫描部分桶
 *          采用先粗筛后精排的两阶段策略，平衡速度和精度
 */
class IVFIndex {
public:
    /**
     * @brief   构造函数
     * @param   dim       向量维度
     * @param   n_lists   桶数量设为数据量的√倍，100万数据用1000桶
     */
    IVFIndex(int dim, int n_lists) 
        : dim_(dim), n_lists_(n_lists), kmeans_(n_lists, 5, dim) {
        inverted_lists_.resize(n_lists);
    }

    /**
     * @brief   构建IVF索引
     * @param   dataset   待索引的向量数据集
     * @note    包含训练KMeans和填充倒排桶两个阶段
     */
    void build(const VectorDataset& dataset) {
        std::cout << "Training IVF centroids..." << std::endl;
        kmeans_.train(dataset);
        
        std::cout << "Populating inverted lists..." << std::endl;
        const auto& centroids = kmeans_.get_centroids();
        
        // 预先计算分配结果，便于并行
        std::vector<int> assignments(dataset.get_count());
        
        // 并行计算归属桶
        #pragma omp parallel for
        for (idx_t i = 0; i < dataset.get_count(); ++i) {
            auto vec = dataset.get_vector(i);
            int best_cluster = 0;
            float min_dist = std::numeric_limits<float>::max();
            
            for (int c = 0; c < n_lists_; ++c) {
                std::span<const float> center(centroids.data() + c * dim_, dim_);
                float dist = l2_distance(vec, center);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }

        // 串行填充vector（这步很快，无需并行）
        for (idx_t i = 0; i < dataset.get_count(); ++i) {
            inverted_lists_[assignments[i]].push_back(i);
        }
    }

    /**
     * @brief   搜索最近邻向量
     * @param   query          查询向量
     * @param   dataset        数据集
     * @param   k              返回结果数量
     * @param   probe_ratio    探测比例（默认0.2，即距离扩大20%内的桶都搜索）
     * @param   max_nprobe     最大探测桶数
     * @param   refinery_factor  精排因子（预选候选数 = k * factor）
     * @return  按距离排序的K个最近邻
     * @note    采用两阶段策略：先粗筛候选，再精排选出最终结果
     */
    std::vector<SearchResult> search(std::span<const float> query, 
                                     const VectorDataset& dataset, 
                                     int k, 
                                     float probe_ratio = 0.2f, 
                                     int max_nprobe = 20,
                                     int refinery_factor = 5) {
        const auto& centroids = kmeans_.get_centroids();
        std::vector<std::pair<float, int>> clusters_scores; 
        
        // 计算查询向量到所有桶中心的距离
        for (int c = 0; c < n_lists_; ++c) {
            std::span<const float> center(centroids.data() + c * dim_, dim_);
            float dist = l2_distance(query, center);
            clusters_scores.push_back({dist, c});
        }
        // 按距离排序，最近的桶排在前面
        std::sort(clusters_scores.begin(), clusters_scores.end());

        // 确定搜索范围
        float best_center_dist = clusters_scores[0].first;
        // 动态阈值：距离最佳桶一定比例内的桶都搜索
        float dist_threshold = best_center_dist * (1.0f + probe_ratio) + 1e-6f;

        // 粗筛 - 从多个桶中收集候选向量
        std::priority_queue<SearchResult> top_candidates;
        size_t candidates_limit = k * refinery_factor;
        
        int probed_count = 0;
        for (const auto& bucket_info : clusters_scores) {
            float center_dist = bucket_info.first;
            int cluster_id = bucket_info.second;

            // 达到最大探测数，或距离超出阈值则停止
            if (probed_count >= max_nprobe) break;
            if (probed_count > 0 && center_dist > dist_threshold) break;

            const auto& bucket = inverted_lists_[cluster_id];
            probed_count++;

            // 遍历桶内所有向量
            for (idx_t vec_id : bucket) {
                auto vec = dataset.get_vector(vec_id);
                float dist = l2_distance(query, vec);

                // 使用最小堆维护Top-K候选
                if (top_candidates.size() < candidates_limit) {
                    top_candidates.push({vec_id, dist});
                } else if (dist < top_candidates.top().distance) {
                    top_candidates.pop();
                    top_candidates.push({vec_id, dist});
                }
            }
        }

        // 精排 - 从候选中选出最终的K个结果
        std::vector<SearchResult> all_candidates;
        while(!top_candidates.empty()) {
            all_candidates.push_back(top_candidates.top());
            top_candidates.pop();
        }
        
        // 按距离升序排序
        std::sort(all_candidates.begin(), all_candidates.end(), [](const SearchResult& a, const SearchResult& b){
            return a.distance < b.distance;
        });

        // 返回前K个结果
        std::vector<SearchResult> results;
        for (size_t i = 0; i < std::min((size_t)k, all_candidates.size()); ++i) {
            results.push_back(all_candidates[i]);
        }
        
        return results;
    }

private:
    int dim_;                              ///< 向量维度
    int n_lists_;                          ///< IVF桶数量
    KMeans kmeans_;                        ///< KMeans聚类器，用于生成桶中心
    std::vector<std::vector<idx_t>> inverted_lists_;  ///< 倒排桶列表，存储向量ID
};

} // namespace minimilvus