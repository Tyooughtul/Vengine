/**
 * @file    test_wal.cpp
 * @brief   WAL 测试
 */

#include <iostream>
#include "../src/core/utils/wal.hpp"

int main() {
    std::cout << "=== WAL Test ===" << std::endl;
    
    // 创建 WAL（如果已有日志，会自动恢复）
    minimilvus::WAL wal("test_wal.log");
    
    // 模拟一些操作
    std::cout << "\nSimulating operations:" << std::endl;
    
    wal.append("ADD_VECTOR", "vector_1: [1.0, 2.0, 3.0]");
    wal.append("ADD_VECTOR", "vector_2: [4.0, 5.0, 6.0]");
    wal.append("ADD_VECTOR", "vector_3: [7.0, 8.0, 9.0]");
    
    std::cout << "Operations recorded." << std::endl;
    
    // 3. 模拟崩溃恢复
    std::cout << "\nSimulating crash and recovery..." << std::endl;
    std::cout << "(Delete the WAL object and create a new one)" << std::endl;
    
    // WAL 超出作用域会自动析构
    // 这里我们模拟重新启动
    {
        // 程序退出，模拟崩溃
    }
    
    // 重新创建 WAL，会触发恢复
    std::cout << "\n=== Restarting ===" << std::endl;
    minimilvus::WAL wal2("test_wal.log");
    
    std::cout << "\nTest completed!" << std::endl;
    
    return 0;
}