#pragma once

#include <atomic>

class Counter {
public:
    Counter() : val_(0) {}

    int operator++(int v) {
        return val_++;
    }

    int get() {
        return val_.load();
    }

private:
    std::atomic<int> val_;
};
