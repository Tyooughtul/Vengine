#pragma once
#include <vector>
#include <stdexcept>
#include <span>

namespace minimilvus {

using idx_t = int64_t;
using scalar_t = float;

class VectorDataset {
public:
    VectorDataset(int dim) : dim_(dim) {}
    
    void add(const std::vector<scalar_t>& vec) {
        if (vec.size() != dim_) throw std::invalid_argument("Dimension Mismatch");
        data_.insert(data_.end(), vec.begin(), vec.end());
        cnt_++;
    }

    std::span<const scalar_t> get_vector(idx_t i) const {
        return {data_.data() + i * dim_, static_cast<size_t>(dim_)};
    }

    int64_t get_dim() const { return dim_; }
    int64_t get_count() const { return cnt_; }
    
private:
    int64_t dim_ = 0;
    int64_t cnt_ = 0;
    std::vector<scalar_t> data_;
};

}