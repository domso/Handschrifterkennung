#pragma once

#include <mutex>
#include <condition_variable>
#include <iostream>
#include <pthread.h>
#include <atomic>

class Barrier1 {
public:

    explicit Barrier1(const int n) {
    	pthread_barrier_init(&m_bar, 0, n);
    }

	Barrier1(const Barrier1& b) = delete;
    void wait() {
    	pthread_barrier_wait(&m_bar);
    }

private:
    pthread_barrier_t m_bar;
};

#pragma once

#include <mutex>
#include <condition_variable>
#include <iostream>
#include <pthread.h>
#include <atomic>
#include <thread>

class Barrier {
public:
	explicit Barrier(const int n) :
			lockCounter_(0), head_(n), size_(n) {
		m_head = n;
		m_counter = 0;
	}

	Barrier(const Barrier& b) = delete;

	void wait() {

		// get current ticket
		int ticket = m_head;
		//std::unique_lock < std::mutex > ul(mutex);


		// increase current number of threads waiting on the barrier
		m_counter++;

		// if all threads reached the wait() call, proceed with the call
		if (m_counter == size_) {
			int tmp = size_;
			if(m_counter.compare_exchange_strong(tmp, 0)) {
				//m_counter = 0;
				// invalidates all current tickets
				m_head = ticket + size_;
			}
			return;
		}

		while (ticket == m_head) {
			std::this_thread::yield();
		}
	}

private:
	std::atomic<int> m_head;
	std::atomic<int> m_counter;
	int lockCounter_;
	int head_;
	const int size_;

	std::mutex mutex;
};
