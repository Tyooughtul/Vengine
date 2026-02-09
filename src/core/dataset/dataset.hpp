/**
 * @file    dataset.hpp
 * @brief   向量数据集管理类
 * @author  Tyooughtul
 */

#pragma once
#include <vector>
#include <stdexcept>
#include <span>

namespace minimilvus {

using std::vector;
using std::span;

using idx_t = int64_t;
using scalar_t = float;

class VectorDataset {
public:
    explicit VectorDataset(int dim) : dim_(dim) {}

    void add(const std::vector<scalar_t>& vec);

    std::span<const scalar_t> get_vector(idx_t i) const;

    int64_t get_dim() const { return dim_; }

    int64_t get_count() const { return cnt_; }
    
private:
    int64_t dim_ = 0;
    int64_t cnt_ = 0;
    std::vector<scalar_t> data_;
};

}
