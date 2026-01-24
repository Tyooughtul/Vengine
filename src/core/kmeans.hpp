#pragma once
// 补充缺失的头文件
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <stdexcept> 
#include "dataset.hpp"
#include "metrics.hpp"

namespace minimilvus {

class KMeans {
public:
    KMeans(int k, int max_iter, int dim) : k_(k), max_iter_(max_iter), dim_(dim) {
        centroids_.resize(k_ * dim_);
    }

    void train(const VectorDataset& dataset) {
        if (dataset.get_count() < k_) {
            throw std::runtime_error("Datasize is smaller than k");
        }

        init_centroids(dataset);
        std::vector<int> assign(dataset.get_count(), 0);

        for (int iter = 0; iter < max_iter_; iter++) {
            bool changed = false;
            for (idx_t i = 0; i < dataset.get_count(); i++) {
                auto vec = dataset.get_vector(i);
                int best_cluster = 0;
                float min_dist = std::numeric_limits<float>::max();
                for (int c = 0; c < k_; c++) {
                    // 从centroids_.data()+c*dim_开始的，长度为dim_的一个连续内存的视图
                    std::span<const float> centroid(centroids_.data() + c * dim_, dim_);
                    float d = l2_distance(vec, centroid); 
                    if (d < min_dist) {
                        min_dist = d;
                        best_cluster = c;
                    }
                }
                if (assign[i] != best_cluster) {
                    assign[i] = best_cluster;
                    changed = true;
                }
            }
            if (!changed && iter > 0) {
                std::cout << "KMeans converged at iteration " << iter << std::endl;
                break;
            }
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
            for (int c = 0; c < k_; c++) {
                if (counts[c] > 0) {
                    for (int d = 0; d < dim_; d++) {
                        new_centroids[c * dim_ + d] /= counts[c];
                    }
                } else {
                    std::cout << "Warning: Empty cluster " << c << " detected." << std::endl;
                    for (int d = 0; d < dim_; ++d) {
                        new_centroids[c * dim_ + d] = centroids_[c * dim_ + d];
                    }
                }
            }
            centroids_ = std::move(new_centroids);
        }
    }

    const std::vector<float>& get_centroids() const {
        return centroids_;
    }

private:
    int k_;
    int max_iter_;
    int dim_;
    std::vector<float> centroids_;

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