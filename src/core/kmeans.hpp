/**
 * @file    kmeans.hpp
 * @brief   KMeans聚类算法实现
 * @details 将向量数据聚类成K个簇，用于IVF索引的桶划分
 * @author  Tyooughtul
 */

#pragma once
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <stdexcept> 
#include <span>
#include <omp.h>
#include "dataset.hpp"
#include "metrics.hpp"

namespace minimilvus {

/**
 * @brief   KMeans聚类算法实现
 * @details 使用经典KMeans算法将向量分成K个簇
 *          采用OpenMP并行化加速训练过程
 */
class KMeans {
public:
    /**
     * @brief   构造函数
     * @param   k           聚类数量（即簇的数量）
     * @param   max_iter    最大迭代次数
     * @param   dim         向量维度
     */
    KMeans(int k, int max_iter, int dim) : k_(k), max_iter_(max_iter), dim_(dim) {
        centroids_.resize(k_ * dim_);
    }

    /**
     * @brief   训练KMeans模型
     * @param   dataset     待聚类的向量数据集
     * @throws  std::runtime_error 当数据量小于K时
     * @note    包含初始化质心和迭代优化两个阶段
     */
    void train(const VectorDataset& dataset) {
        if (dataset.get_count() < k_) {
            throw std::runtime_error("Datasize is smaller than k");
        }

        init_centroids(dataset);
        std::vector<int> assign(dataset.get_count(), 0);

        for (int iter = 0; iter < max_iter_; iter++) {
            int changed_count = 0;

            // 并行化分配步骤
            // 每个向量找到最近的质心，并统计变化的向量数量
            #pragma omp parallel for reduction(+:changed_count)
            for (idx_t i = 0; i < dataset.get_count(); i++) {
                auto vec = dataset.get_vector(i);
                int best_cluster = 0;
                float min_dist = std::numeric_limits<float>::max();
                
                // 扫描所有质心寻找最近的
                for (int c = 0; c < k_; c++) {
                    std::span<const float> centroid(centroids_.data() + c * dim_, dim_);
                    float d = l2_distance(vec, centroid);
                    if (d < min_dist) {
                        min_dist = d;
                        best_cluster = c;
                    }
                }
                
                if (assign[i] != best_cluster) {
                    assign[i] = best_cluster;
                    changed_count++;
                }
            }

            // 当没有向量切换簇时，提前结束
            if (changed_count == 0 && iter > 0) {
                std::cout << "KMeans converged at iteration " << iter << std::endl;
                break;
            }

            // 计算每个簇的向量均值
            std::vector<float> new_centroids(k_ * dim_, 0.0f);
            std::vector<int> counts(k_, 0);
            
            for (idx_t i = 0; i < dataset.get_count(); i++) {
                int cluster_id = assign[i];
                auto vec = dataset.get_vector(i);
                for (int d = 0; d < dim_; d++) {
                    new_centroids[cluster_id * dim_ + d] += vec[d];
                }
                counts[cluster_id]++;
            }

            // 簇中心 = 簇内向量和 / 簇内向量数
            for (int c = 0; c < k_; c++) {
                if (counts[c] > 0) {
                    float inv_count = 1.0f / counts[c];
                    for (int d = 0; d < dim_; d++) {
                        new_centroids[c * dim_ + d] *= inv_count;
                    }
                }
            }
            centroids_ = std::move(new_centroids);
            
            // 打印训练进度
            if (iter % 2 == 0) std::cout << "KMeans iter " << iter << "/" << max_iter_ << "..." << std::endl;
        }
    }

    /**
     * @brief   获取聚类中心
     * @return  聚类中心向量数组，按行优先存储
     */
    const std::vector<float>& get_centroids() const {
        return centroids_;
    }

private:
    int k_;                    ///< 聚类数量
    int max_iter_;             ///< 最大迭代次数
    int dim_;                  ///< 向量维度
    std::vector<float> centroids_;  ///< 聚类中心向量

    /**
     * @brief   初始化质心
     * @param   dataset     数据集
     * @note    使用随机采样策略，从数据集中随机选择K个向量作为初始质心
     */
    void init_centroids(const VectorDataset& dataset) {
        static std::mt19937 rng(42);
        std::uniform_int_distribution<idx_t> dist(0, dataset.get_count() - 1);
        for (int i = 0; i < k_; i++) {
            idx_t rand_idx = dist(rng);
            auto vec = dataset.get_vector(rand_idx);
            std::copy(vec.begin(), vec.end(), centroids_.begin() + i * dim_);
        }
    }
};

} // namespace minimilvus