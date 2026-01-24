#pragma once
#include <span>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <immintrin.h>

namespace minimilvus {
    inline float l2_distance(std::span<const float> a, std::span<const float> b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        
        float sum = 0;
        size_t n = a.size();
        size_t i = 0;
        
        // SIMD path for AVX2
        #ifdef __AVX2__
            const float* a_ptr = a.data();
            const float* b_ptr = b.data();
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (; i + 8 <= n; i += 8) {
                __m256 va = _mm256_loadu_ps(a_ptr + i);
                __m256 vb = _mm256_loadu_ps(b_ptr + i);
                __m256 diff = _mm256_sub_ps(va, vb);
                #ifdef __FMA__
                    sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
                #else
                    __m256 sq = _mm256_mul_ps(diff, diff);
                    sum_vec = _mm256_add_ps(sum_vec, sq);
                #endif
            }
            
            // Horizontal sum
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            for (int k = 0; k < 8; ++k) sum += temp[k];
        #endif

        // Scalar path for remaining elements (or if AVX2 not available)
        for(; i < n; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
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