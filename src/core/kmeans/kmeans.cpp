/**
 * @file    kmeans.cpp
 * @author  Tyooughtul
 */

#include "kmeans.hpp"

namespace minimilvus {

using std::cout;
using std::endl;
using std::runtime_error;
using std::invalid_argument;

void KMeans::train(const VectorDataset& dataset) {
    if (dataset.get_count() < k_) {
        throw std::runtime_error("Datasize is smaller than k");
    }

    init_centroids(dataset);
    std::vector<int> assign(dataset.get_count(), 0);

    for (int iter = 0; iter < max_iter_; iter++) {
        int changed_count = 0;

        #pragma omp parallel for reduction(+:changed_count)
        for (idx_t i = 0; i < dataset.get_count(); i++) {
            auto vec = dataset.get_vector(i);
            int best_cluster = 0;
            float min_dist = std::numeric_limits<float>::max();
            
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

        if (changed_count == 0 && iter > 0) {
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
                float inv_count = 1.0f / counts[c];
                for (int d = 0; d < dim_; d++) {
                    new_centroids[c * dim_ + d] *= inv_count;
                }
            }
        }
        centroids_ = std::move(new_centroids);
        
        if (iter % 2 == 0) std::cout << "KMeans iter " << iter << "/" << max_iter_ << "..." << std::endl;
    }
}

void KMeans::init_centroids(const VectorDataset& dataset) {
    static std::mt19937 rng(42);
    std::uniform_int_distribution<idx_t> dist(0, dataset.get_count() - 1);
    for (int i = 0; i < k_; i++) {
        idx_t rand_idx = dist(rng);
        auto vec = dataset.get_vector(rand_idx);
        std::copy(vec.begin(), vec.end(), centroids_.begin() + i * dim_);
    }
}

} // namespace minimilvus
