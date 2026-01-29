/**
 * @file    metrics.hpp
 * @brief   向量距离度量函数实现
 * @details 提供L2距离和内积距离的计算，支持SIMD加速
 * @author  Tyooughtul
 */

#pragma once
#include <span>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <immintrin.h>

namespace minimilvus {

/**
 * @brief   计算两个向量的L2欧氏距离
 * @param   a  第一个向量
 * @param   b  第二个向量
 * @return   两向量之间的L2距离平方值
 * @throws   std::invalid_argument 当两向量维度不同时
 * @note     采用AVX2 SIMD指令优化，一次处理8个float
 */
inline float l2_distance(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Dimension mismatch");
    }
    
    float sum = 0;
    size_t n = a.size();
    size_t i = 0;
    
    // SIMD path for AVX2 - 一次处理8个float
    #ifdef __AVX2__
        const float* a_ptr = a.data();
        const float* b_ptr = b.data();
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a_ptr + i);
            __m256 vb = _mm256_loadu_ps(b_ptr + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            #ifdef __FMA__
                // 合并乘法和加法操作
                sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
            #else
                __m256 sq = _mm256_mul_ps(diff, diff);
                sum_vec = _mm256_add_ps(sum_vec, sq);
            #endif
        }
        
        // 将8个累加结果合并为一个值
        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);
        for (int k = 0; k < 8; ++k) sum += temp[k];
    #endif

    // 处理剩余元素或无SIMD支持的情况
    for(; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * @brief   计算两个向量的内积
 * @param   a  第一个向量
 * @param   b  第二个向量
 * @return   两向量的内积值
 * @throws   std::invalid_argument 当两向量维度不同时
 */
inline float ip_distance(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Dimension mismatch");
    }
    float sum = 0;
    for(size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
}