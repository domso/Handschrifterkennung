#include "logger.h"

bool logger::has_error() const {
	return m_error;
}

void logger::clear() {
	m_error = false;
	m_messageBuffer.clear();
}

void logger::print_all() const {
	for (const message& m : m_messageBuffer) {
		if (m.error) {
			std::cerr << m.text << std::endl;
		} else {
			std::cout << m.text << std::endl;
		}
	}
}

