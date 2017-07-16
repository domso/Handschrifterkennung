#pragma once

#include <mutex>
#include <condition_variable>
#include <iostream>
#include <pthread.h>

class Barrier {
public:
    explicit Barrier(const int n) {
    	pthread_barrier_init(&m_bar, 0, n);
    }

    ~Barrier() {
    	pthread_barrier_destroy(&m_bar);
    }

    Barrier(const Barrier& b) = delete;

    void wait() {
    	pthread_barrier_wait(&m_bar);
    }

private:
    pthread_barrier_t m_bar;
};
