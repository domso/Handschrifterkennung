#pragma once

#include <mutex>
#include <condition_variable>
#include <iostream>

class Barrier {
public:
    explicit Barrier(const int n): lockCounter_(0), head_(n), size_(n) {

    }

    void wait() {
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
        }

        // wait until current ticket is invalid
        while (ticket == head_) {
            cond.wait(ul);
        }

        // notify all other waiting threads
        cond.notify_all();
    }

private:
    int lockCounter_;
    int head_;
    int size_;

    std::mutex mutex;
    std::condition_variable cond;
};
