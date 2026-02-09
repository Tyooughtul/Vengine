/**
 * @file    kmeans.hpp
 * @brief   KMeans聚类算法
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

using std::vector;
using std::span;
using std::cout;
using std::endl;
using std::runtime_error;
using std::invalid_argument;

class KMeans {
public:
    KMeans(int k, int max_iter, int dim) : k_(k), max_iter_(max_iter), dim_(dim) {
        centroids_.resize(k_ * dim_);
    }

    void train(const VectorDataset& dataset);

    const std::vector<float>& get_centroids() const {
        return centroids_;
    }

private:
    int k_;
    int max_iter_;
    int dim_;
    std::vector<float> centroids_;

    void init_centroids(const VectorDataset& dataset);
};

} // namespace minimilvus
