#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>

template <typename callableT>
class Thread_Pool {
public:

    // initializes the thread-pool with numThreads threads
    explicit Thread_Pool(const int numThreads) : nextThread_(0), runningTasks_(0), threads_(numThreads), threadContext_(numThreads) {
        for (size_t i = 0; i < threadContext_.size(); i++) {
            threadContext_[i].runnning_.store(true);
        }

        for (size_t i = 0; i < threads_.size(); i++) {
            threads_[i] = std::thread(&Thread_Pool::threadMain, this, i);
        }
    }

    ~Thread_Pool() {
        for (size_t i = 0; i < threadContext_.size(); i++) {
            threadContext_[i].runnning_.store(false);
            threadContext_[i].cond_.notify_all();
        }

        for (size_t i = 0; i < threads_.size(); i++) {
            if (threads_[i].joinable()) {
                threads_[i].join();
            }
        }
    }

    // closes the thread-pool
    void close() {
        for (size_t i = 0; i < threadContext_.size(); i++) {
            threadContext_[i].runnning_.store(false);
            threadContext_[i].cond_.notify_all();
        }
    }

    // inserts n callable-Tasks into the local queue of the next selected thread
    // load balancing is done by round-robin
    void insert(callableT& task, const int n = 1) {
        runningTasks_.fetch_add(1);

        int id = nextThread_.fetch_add(1) % threadContext_.size();

        std::unique_lock<std::mutex> ul(threadContext_[id].mutex_);

        for (int i = 0; i < n; i++) {
            threadContext_[id].queue_.push(task);
        }

        threadContext_[id].cond_.notify_all();
    }

    // waits until all tasks are completed
    void wait() {
        std::unique_lock<std::mutex> ul(CloseMutex_);

        while (runningTasks_.load() != 0) {
            CloseCond_.wait(ul);
        }

    }

private:
    // internal thread-function
    // id represents the index in the local threadContext-vector ([0, numThreads])
    void threadMain(const int id) {
        while (true) {
            std::unique_lock<std::mutex> ul(threadContext_[id].mutex_);

            // while queue is empty
            while (threadContext_[id].queue_.empty() && threadContext_[id].runnning_) {
                threadContext_[id].cond_.wait(ul);
            }

            if (!threadContext_[id].runnning_) {
                return;
            }

            // get next task
            callableT task = threadContext_[id].queue_.front();
            threadContext_[id].queue_.pop();
            ul.unlock(); // <- unlock the mutex here, so the task can safely access the insert()-call to generate new tasks

            // the task needs the thread-pool as argument
            task(this);

            // set the task as finished
            runningTasks_.fetch_sub(1);

            if (runningTasks_.load() == 0) {
                CloseCond_.notify_all();
            }
        }
    }

    // local structure for a worker-thread
    struct local_thread_context_t {
        std::atomic<bool> runnning_;
        std::mutex mutex_;
        std::condition_variable cond_;
        std::queue<callableT> queue_;
    };

    std::atomic<int> nextThread_;

    std::atomic<int> runningTasks_;
    std::vector<std::thread> threads_;
    std::vector<local_thread_context_t> threadContext_;

    std::mutex CloseMutex_;
    std::condition_variable CloseCond_;


};


