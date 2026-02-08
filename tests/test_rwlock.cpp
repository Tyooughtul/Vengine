/**
 * @file    test_rwlock.cpp
 * @brief   读写锁测试
 */

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include "../src/core/utils/rwlock.hpp"

// 共享数据
int shared_data = 0;

// 选择使用哪种实现
// minimilvus::ManualRWLock rwlock;  // 手写实现
minimilvus::StdRWLock rwlock;      // 标准库实现

/**
 * @brief   读者线程
 */
void reader_thread(int id) {
    for (int i = 0; i < 5; ++i) {
        minimilvus::StdRWLock::ReadLock lock(rwlock);
        
        int value = shared_data;
        std::cout << "Reader " << id << " reads: " << value << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

/**
 * @brief   写者线程
 */
void writer_thread(int id) {
    for (int i = 0; i < 3; ++i) {
        minimilvus::StdRWLock::WriteLock lock(rwlock);
        
        shared_data++;
        std::cout << "Writer " << id << " writes: " << shared_data << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

int main() {
    std::cout << "=== RWLock Test ===" << std::endl;
    
    // 创建3个读者线程
    std::vector<std::thread> readers;
    for (int i = 0; i < 3; ++i) {
        readers.emplace_back(reader_thread, i);
    }
    
    // 创建1个写者线程
    std::vector<std::thread> writers;
    for (int i = 0; i < 1; ++i) {
        writers.emplace_back(writer_thread, i);
    }
    
    // 等待所有线程完成
    for (auto& t : readers) {
        t.join();
    }
    for (auto& t : writers) {
        t.join();
    }
    
    std::cout << "All threads completed!" << std::endl;
    std::cout << "Final value: " << shared_data << std::endl;
    
    return 0;
}