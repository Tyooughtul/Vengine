/**
 * @file    rwlock.hpp
 * @brief   读写锁实现（提供两种方式）
 * @details 方式1：手写实现（学习原理）
 *          方式2：封装 std::shared_mutex（生产使用）
 * @author  Tyooughtul
 */

#pragma once

#include <mutex>
#include <condition_variable>
#include <shared_mutex>

namespace minimilvus {

// ============================================================
// 方式1：手写读写锁（学习原理）
// ============================================================

/**
 * @brief   手写读写锁类
 * @details 使用条件变量和互斥锁实现读写锁逻辑
 *          适合学习原理，理解读写锁的工作机制
 */
class ManualRWLock {
public:
    /**
     * @brief   构造函数
     */
    ManualRWLock() : readers_(0), writers_(0), write_requested_(false) {}
    
    // 禁止拷贝
    ManualRWLock(const ManualRWLock&) = delete;
    ManualRWLock& operator=(const ManualRWLock&) = delete;
    
    /**
     * @brief   获取读锁
     * @note    如果有写锁或写请求，会阻塞
     */
    void lock_read() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 等待：没有写锁，且没有写请求
        cv_read_.wait(lock, [this] {
            return !write_requested_ && writers_ == 0;
        });
        
        // 增加读者数量
        readers_++;
    }
    
    /**
     * @brief   释放读锁
     */
    void unlock_read() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 减少读者数量
        readers_--;
        
        // 如果没有读者了，通知等待的写者
        if (readers_ == 0) {
            cv_write_.notify_one();
        }
    }
    
    /**
     * @brief   获取写锁
     * @note    会阻塞所有读者和写者
     */
    void lock_write() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 标记有写请求
        write_requested_ = true;
        
        // 等待：没有读者，且没有其他写者
        cv_write_.wait(lock, [this] {
            return readers_ == 0 && writers_ == 0;
        });
        
        // 增加写者数量
        writers_++;
    }
    
    /**
     * @brief   释放写锁
     */
    void unlock_write() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // 减少写者数量
        writers_--;
        
        // 取消写请求标记
        write_requested_ = false;
        
        // 通知所有等待的读者
        cv_read_.notify_all();
        
        // 通知等待的写者（如果有）
        if (writers_ == 0) {
            cv_write_.notify_one();
        }
    }
    
    /**
     * @brief   读锁的 RAII 包装
     */
    class ReadLock {
    public:
        explicit ReadLock(ManualRWLock& lock) : lock_(lock) {
            lock_.lock_read();
        }
        
        ~ReadLock() {
            lock_.unlock_read();
        }
        
        ReadLock(const ReadLock&) = delete;
        ReadLock& operator=(const ReadLock&) = delete;
        
    private:
        ManualRWLock& lock_;
    };
    
    /**
     * @brief   写锁的 RAII 包装
     */
    class WriteLock {
    public:
        explicit WriteLock(ManualRWLock& lock) : lock_(lock) {
            lock_.lock_write();
        }
        
        ~WriteLock() {
            lock_.unlock_write();
        }
        
        WriteLock(const WriteLock&) = delete;
        WriteLock& operator=(const WriteLock&) = delete;
        
    private:
        ManualRWLock& lock_;
    };
    
private:
    mutable std::mutex mutex_;              ///< 保护内部状态的互斥锁
    std::condition_variable cv_read_;         ///< 读者条件变量
    std::condition_variable cv_write_;        ///< 写者条件变量
    
    int readers_;                         ///< 当前读者数量
    int writers_;                         ///< 当前写者数量（0或1）
    bool write_requested_;                  ///< 是否有写请求
};


// ============================================================
// 方式2：封装 std::shared_mutex（生产使用）
// ============================================================

/**
 * @brief   读写锁类（基于 C++17 std::shared_mutex）
 * @details 封装标准库的读写锁，提供更清晰的接口
 *          适合生产环境使用
 */
class StdRWLock {
public:
    /**
     * @brief   构造函数
     */
    StdRWLock() = default;
    
    // 禁止拷贝
    StdRWLock(const StdRWLock&) = delete;
    StdRWLock& operator=(const StdRWLock&) = delete;
    
    /**
     * @brief   获取读锁
     */
    void lock_read() {
        mtx_.lock_shared();
    }
    
    /**
     * @brief   释放读锁
     */
    void unlock_read() {
        mtx_.unlock_shared();
    }
    
    /**
     * @brief   获取写锁
     */
    void lock_write() {
        mtx_.lock();
    }
    
    /**
     * @brief   释放写锁
     */
    void unlock_write() {
        mtx_.unlock();
    }
    
    /**
     * @brief   读锁的 RAII 包装
     */
    class ReadLock {
    public:
        explicit ReadLock(StdRWLock& lock) : lock_(lock) {
            lock_.lock_read();
        }
        
        ~ReadLock() {
            lock_.unlock_read();
        }
        
        ReadLock(const ReadLock&) = delete;
        ReadLock& operator=(const ReadLock&) = delete;
        
    private:
        StdRWLock& lock_;
    };
    
    /**
     * @brief   写锁的 RAII 包装
     */
    class WriteLock {
    public:
        explicit WriteLock(StdRWLock& lock) : lock_(lock) {
            lock_.lock_write();
        }
        
        ~WriteLock() {
            lock_.unlock_write();
        }
        
        WriteLock(const WriteLock&) = delete;
        WriteLock& operator=(const WriteLock&) = delete;
        
    private:
        StdRWLock& lock_;
    };
    
private:
    std::shared_mutex mtx_;  ///< C++17 标准库的读写锁
};

}  // namespace minimilvus