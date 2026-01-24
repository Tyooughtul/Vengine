#pragma once
#include <vector>
#include <algorithm>
#include <queue>
#include "kmeans.hpp"
#include "dataset.hpp"
#include "metrics.hpp"

namespace minimilvus {

struct SearchResult {
    idx_t id;
    float distance;
    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }
};

class IVFIndex {
public:
    IVFIndex(int dim, int n_lists) 
        : dim_(dim), n_lists_(n_lists), kmeans_(n_lists, 20, dim) {
        inverted_lists_.resize(n_lists);
    }
    void build(const VectorDataset& dataset) {
        std::cout << "Training IVF centroids..." << std::endl;
        kmeans_.train(dataset);
        std::cout << "Populating inverted lists..." << std::endl;
        const auto& centroids = kmeans_.get_centroids();
        for (idx_t i = 0; i < dataset.get_count(); ++i) {
            auto vec = dataset.get_vector(i);
            int best_cluster = -1;
            float min_dist = std::numeric_limits<float>::max();

            for (int c = 0; c < n_lists_; ++c) {
                std::span<const float> center(centroids.data() + c * dim_, dim_);
                float dist = l2_distance(vec, center);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            inverted_lists_[best_cluster].push_back(i);
        }
    }
    std::vector<SearchResult> search(std::span<const float> query, 
                                     const VectorDataset& dataset, 
                                     int k, int nprobe) {
        const auto& centroids = kmeans_.get_centroids();
        std::vector<std::pair<float, int>> clusters_scores; // (dist, cluster_id)
        
        for (int c = 0; c < n_lists_; ++c) {
            std::span<const float> center(centroids.data() + c * dim_, dim_);
            float dist = l2_distance(query, center);
            clusters_scores.push_back({dist, c});
        }
        std::sort(clusters_scores.begin(), clusters_scores.end());

        std::priority_queue<SearchResult> top_candidates;

        for (int i = 0; i < nprobe && i < n_lists_; ++i) {
            int cluster_id = clusters_scores[i].second;
            const auto& bucket = inverted_lists_[cluster_id];

            for (idx_t vec_id : bucket) {
                auto vec = dataset.get_vector(vec_id);
                float dist = l2_distance(query, vec);

                if (top_candidates.size() < k) {
                    top_candidates.push({vec_id, dist});
                } else if (dist < top_candidates.top().distance) {
                    top_candidates.pop();
                    top_candidates.push({vec_id, dist});
                }
            }
        }

        std::vector<SearchResult> results;
        while (!top_candidates.empty()) {
            results.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

private:
    int dim_;
    int n_lists_;
    KMeans kmeans_;
    std::vector<std::vector<idx_t>> inverted_lists_;
};

} // namespace minimilvus