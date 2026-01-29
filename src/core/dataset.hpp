/**
 * @file    dataset.hpp
 * @brief   向量数据集管理类实现
 * @details 提供高维向量的存储、添加和随机访问功能
 * @author  yourname
 * @date    2025-01-29
 */

#pragma once
#include <vector>
#include <stdexcept>
#include <span>

namespace minimilvus {

/// 向量索引类型，用于标识数据集中的向量
using idx_t = int64_t;

/// 向量元素类型，当前使用float单精度浮点数
using scalar_t = float;

/**
 * @brief   向量数据集管理类
 * @details 负责存储和访问高维向量数据，支持批量添加和随机访问
 *          向量以扁平化方式存储在连续内存中，以提高缓存命中率
 */
class VectorDataset {
public:
    /**
     * @brief   构造函数
     * @param   dim     向量的维度
     * @note    所有向量的维度必须一致
     */
    explicit VectorDataset(int dim) : dim_(dim) {}

    /**
     * @brief   向数据集中添加一个向量
     * @param   vec     待添加的向量，维度必须与数据集维度一致
     * @throws  std::invalid_argument 当向量维度不匹配时
     */
    void add(const std::vector<scalar_t>& vec) {
        if (vec.size() != dim_) throw std::invalid_argument("Dimension Mismatch");
        data_.insert(data_.end(), vec.begin(), vec.end());
        cnt_++;
    }

    /**
     * @brief   获取指定索引的向量
     * @param   i       向量索引
     * @return  该向量的只读视图，避免数据拷贝
     */
    std::span<const scalar_t> get_vector(idx_t i) const {
        return {data_.data() + i * dim_, static_cast<size_t>(dim_)};
    }

    /**
     * @brief   获取向量维度
     * @return  向量的维度
     */
    int64_t get_dim() const { return dim_; }

    /**
     * @brief   获取数据集中向量的数量
     * @return  向量总数
     */
    int64_t get_count() const { return cnt_; }
    
private:
    int64_t dim_ = 0;      ///< 向量维度
    int64_t cnt_ = 0;      ///< 向量数量
    std::vector<scalar_t> data_;  ///< 扁平化存储的向量数据
};

}