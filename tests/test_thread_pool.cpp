/**
 * @file    test_thread_pool.cpp
 * @brief   线程池测试
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <future>
#include "../src/core/utils/thread_pool.hpp"

void print_hello(int id) {
    std::cout << "Thread " << id << " says hello!" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

int main() {
    std::cout << "=== ThreadPool Test ===" << std::endl;
    
    // 创建线程池（4个线程）
    minimilvus::ThreadPool pool(4);
    std::cout << "Created pool with " << pool.num_threads() << " threads" << std::endl;
    
    // 提交8个任务（线程只有4个，所以会并行执行）
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < 8; ++i) {
        // 使用 lambda 捕获 i
        auto fut = pool.submit([i]() {
            print_hello(i);
        });
        futures.push_back(std::move(fut));
    }
    
    std::cout << "Submitted 8 tasks" << std::endl;
    
    // 等待所有任务完成
    // get() 会阻塞直到任务完成
    for (auto& fut : futures) {
        fut.get();
    }
    
    std::cout << "All tasks completed!" << std::endl;
    
    return 0;
}