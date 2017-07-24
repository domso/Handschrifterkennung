#include "barrier.hpp"

namespace util {

barrier::barrier(const int n) :
		m_head(n), m_counter(0), m_size(n) {

}

void barrier::wait() {
	// get current ticket
	int ticket = m_head;

	// increase current number of threads waiting on the barrier
	int current = m_counter++;

	// if all threads reached the wait() call, proceed with the call
	if (current + 1 == m_size) {
		int tmp = m_size;
		if (m_counter.compare_exchange_strong(tmp, 0)) {
			// invalidates all current tickets
			m_head = ticket + m_size;
		}
		return;
	}

	while (ticket == m_head) {
		std::this_thread::yield();
	}
}
}
