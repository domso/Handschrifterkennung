#pragma once

#include <mutex>
#include <condition_variable>
#include <iostream>
#include <pthread.h>

class Barrier {
public:
    explicit Barrier(const int n): lockCounter_(0), head_(n), size_(n) {
    	pthread_barrier_init(&m_bar, 0, n);
    }

    ~Barrier() {
    	pthread_barrier_destroy(&m_bar);
    }

    Barrier(const Barrier& b) = delete;

    void wait() {
    	pthread_barrier_wait(&m_bar);
    	return;
        std::unique_lock<std::mutex> ul(mutex);

        // get current ticket
        int ticket = head_;

        // increase current number of threads waiting on the barrier
        lockCounter_++;

        // if all threads reached the wait() call, proceed with the call
        if (lockCounter_ == size_) {
            lockCounter_ = 0;
            // invalidates all current tickets
            head_ += size_;

            // notify all other waiting threads
            cond.notify_all();

            return;
        }

        // wait until current ticket is invalid
        while (ticket == head_) {
            cond.wait(ul);
        }
    }

private:
    int lockCounter_;
    int head_;
    int size_;

    pthread_barrier_t m_bar;
    std::mutex mutex;
    std::condition_variable cond;
};
