#include "logger.h"

bool logger::has_error() const {
	return error_;
}

void logger::clear() {
	error_ = false;
	message_buffer_.clear();
}

void logger::print_all() const {
	for (const message& m : message_buffer_) {
		if (m.error) {
			std::cerr << m.text << std::endl;
		} else {
			std::cout << m.text << std::endl;
		}
	}
}

