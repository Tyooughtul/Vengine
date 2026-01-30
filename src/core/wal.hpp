/**
 * @file    wal.hpp
 * @brief   Write-Ahead Log 实现（简化版）
 * @details 先写日志再修改数据，保证数据持久性
 * @author  Tyooughtul
 */

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <mutex>
#include <iostream>

namespace minimilvus {

/**
 * @brief   WAL（Write-Ahead Log）类
 * @details 简化版本：使用文本格式，便于理解
 */
class WAL {
public:
    /**
     * @brief   构造函数
     * @param   log_file_path   日志文件路径
     */
    explicit WAL(const std::string& log_file_path) : log_file_path_(log_file_path) {
        // 尝试从日志恢复数据
        recover();
    }
    
    /**
     * @brief   析构函数
     */
    ~WAL() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
    
    // 禁止拷贝
    WAL(const WAL&) = delete;
    WAL& operator=(const WAL&) = delete;
    
    /**
     * @brief   追加日志
     * @param   operation   操作类型（如 "ADD_VECTOR"）
     * @param   data        数据内容
     * @return  是否成功
     */
    bool append(const std::string& operation, const std::string& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 1. 打开文件（追加模式）
        std::ofstream file(log_file_path_, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open WAL file" << std::endl;
            return false;
        }
        
        // 2. 写入日志格式：OPERATION|DATA\n
        file << operation << "|" << data << "\n";
        
        // 3. 刷盘（确保数据持久化）
        file.flush();
        file.close();
        
        return true;
    }
    
    /**
     * @brief   清空日志（检查点）
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::ofstream file(log_file_path_, std::ios::trunc);
        file.close();
        
        std::cout << "WAL cleared (checkpoint)" << std::endl;
    }
    
private:
    std::string log_file_path_;      ///< 日志文件路径
    std::fstream log_file_;         ///< 日志文件流
    mutable std::mutex mutex_;      ///< 保护文件操作
    
    /**
     * @brief   从日志恢复
     */
    void recover() {
        // 1. 尝试打开日志文件
        std::ifstream file(log_file_path_);
        if (!file.is_open()) {
            // 文件不存在，无需恢复
            return;
        }
        
        std::cout << "=== Recovering from WAL ===" << std::endl;
        
        // 2. 逐行读取日志
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            // 3. 解析日志行：OPERATION|DATA
            size_t pos = line.find('|');
            if (pos == std::string::npos) continue;
            
            std::string op = line.substr(0, pos);
            std::string data = line.substr(pos + 1);
            
            // 4. 重放操作
            replay_operation(op, data);
        }
        
        std::cout << "=== Recovery completed ===" << std::endl;
    }
    
    /**
     * @brief   重放操作
     * @param   op      操作类型
     * @param   data    数据
     */
    void replay_operation(const std::string& op, const std::string& data) {
        if (op == "ADD_VECTOR") {
            std::cout << "  Replay: ADD_VECTOR -> " << data << std::endl;
        } else if (op == "DELETE_VECTOR") {
            std::cout << "  Replay: DELETE_VECTOR -> " << data << std::endl;
        }
    }
};

}  // namespace minimilvus