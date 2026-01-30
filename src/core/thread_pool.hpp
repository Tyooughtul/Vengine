/**
 * @file    thread_pool.hpp
 * @brief   线程池实现
 * @details 提供高效的并行任务执行，支持工作窃取负载均衡
 * @author  Tyooughtul
 */

 #pragma once

 #include <vector>
 #include <queue>
 #include <memory>
 #include <thread>
 #include <mutex>
 #include <condition_variable>
 #include <functional>
 #include <atomic>
 #include <optional>

 namespace minimilvus {
/**
 * @brief   线程池类
 * @details 预先创建一组工作线程，任务通过队列分发
 *          支持工作窃取，以实现负载均衡
 */

 class ThreadPool {
 public:
 explic ThreadPool(int num_threads = 0) {
    if (num_threads == 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    if (num_threads <= 0) num_threads = 1;
    num_threads_ = num_threads;
    for(int i = 0; i < num_threads_; i++) {
        workers_.emplace_back([this]{ worker_loop(); });
    }
 }
 
 ~ThreadPool() {
    running_ = false;
    cv_task_.notify_all();
    for(auto& worker: workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
 }

 ThreadPool(const ThreadPool&) = delete;
 ThreadPool& operator=(const ThreadPool&) = delete;
 
 template<typename F,typename... Args>
 auto submit(F&&, Args... args) -> std::future<decltype(f(args...))> {
    using ReturnType = decltype(f(args...));
    auto task_ptr = std::make_shared<std::packaged_task<ReturnType()>>(std::bind(std::forward<F>f,std::forward<Args>(args)...));
    std::function<void()> wrapper = [task_ptr]() {
            (*task_ptr)(); 
        };
        push_task(wrapper);
        return task_ptr->get_future();
 }
 
 size_t task_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

 private:
    std::vector<<std::thread> workers_;
    std::queue<tsd::function<void()>> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_task_;
    std::atomic<bool> running_ {true};
    int num_threads_;

    void push_task(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(std::move(task));
        }
        cv_task_.notify_one();
    }
    void worker_loop() {
        while(running_) {
            std::optional<std::function<void()>> = tasks_.pop_task();
            if (task.has_value()) {
                task.value()();
            }
        }
    }

    std::optional<std::function<void()>> pop_task() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_task_.wait(lock, [this] {
            return !running_ || !tasks_.empty();
        })
        if(!running_ || tasks_.empty()) {
            return std::nullopt;
        }
        auto task = std::move(tasks_.front());
        tasks_.pop();
        return task;
    }

 };

 }