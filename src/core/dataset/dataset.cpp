/**
 * @file    dataset.cpp
 * @author  Tyooughtul
 */

#include "dataset.hpp"

namespace minimilvus {

void VectorDataset::add(const std::vector<float>& vec) {
    if (vec.size() != dim_) throw std::invalid_argument("Dimension Mismatch");
    data_.insert(data_.end(), vec.begin(), vec.end());
    cnt_++;
}

std::span<const float> VectorDataset::get_vector(idx_t i) const {
    return {data_.data() + i * dim_, static_cast<size_t>(dim_)};
}

} // namespace minimilvus
