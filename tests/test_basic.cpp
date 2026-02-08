#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../src/core/dataset/dataset.hpp"
#include "../src/core/metrics/metrics.hpp"

bool is_close(float a,float b,float epsilon = 1e-5) {
    return std::abs(a-b) < epsilon;
}

int main() {
    std::cout << "Running Basic Test" <<std::endl;
    minimilvus::VectorDataset dataset(3);
    dataset.add({1.0,2.0,3.0});
    dataset.add({4.0,5.0,6.0});

    assert(dataset.get_dim() == 3);
    assert(dataset.get_count() == 2);

    // Test data load and store
    auto vec0 = dataset.get_vector(0);
    assert(is_close(vec0[0], 1.0));
    assert(is_close(vec0[2], 3.0));

    // Test L2 Distance
    // vec0: [1, 2, 3], vec1: [4, 5, 6]
    // diff: [3, 3, 3] -> sq: [9, 9, 9] -> sum: 27
    auto vec1 = dataset.get_vector(1);
    float dist = minimilvus::l2_distance(vec0, vec1);
    std::cout << "L2 Distance: " << dist << std::endl;
    assert(is_close(dist, 27.0));

    // Test Inner Product
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    float ip = minimilvus::ip_distance(vec0, vec1);
    std::cout << "IP Distance: " << ip << std::endl;
    assert(is_close(ip, 32.0));

    std::cout << "ALL TESTS PASSED! 🚀" << std::endl;
    return 0;
}