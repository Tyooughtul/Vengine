/**
 * @file    core.cpp
 * @brief   Mini-Milvus核心模块入口
 * @details 提供版本信息等基础功能
 * @author  yourname
 * @date    2025-01-29
 */

#include <iostream>

namespace minimilvus {

/**
 * @brief   打印版本信息
 * @details 在程序启动时调用，用于确认库版本
 */
void print_version() {
    std::cout << "Mini-Milvus Core v0.1.0" << std::endl;
}

} // namespace minimilvus