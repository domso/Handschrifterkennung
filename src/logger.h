#ifndef logger_h
#define logger_h

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <ostream>
/*
 example:

struct runtime_error {
	constexpr static int id = 0;             // ID for identification with logger::check() and logger::print_only()
	constexpr static bool critical = false;  // abort execution in logger::log()
	constexpr static bool error = true; 	 // just a warning / info?
	constexpr static bool date = false;		 // add the current date
	constexpr static bool print = true;	     // print with every logger::log() call
	constexpr static bool hide = false;		 // do not store the message in the logger-buffer
	constexpr static const auto text = "[Error] Could not do something!";
};

void test() {
	logger l;

	l.log<runtime_error>("optional argument");
	// output in std::cerr: [Error] Could not do something! [optional argument]

	if (l.has_error()) { // true, because of runtime_error::error == true
		if (l.check<runtime_error>()) { // true, because of runtime_error::hide == false
			l.print_only<runtime_error>(); // not very efficient here
			// output in std::cerr: [Error] Could not do something! [optional argument]
		}
	}

	l.clear(); // clear all messages
}


 */
class logger {
public:
	template<typename T>
	void log(const std::string argument = "") {
		insert<T>(argument);

		if (T::error) {
			error_ = true;
		}

		if (T::critical) {
			print_all();

			std::cerr << "Execution reached critical error!" << std::endl;
			std::abort();
		}
	}

	template<typename T>
	bool check() const {
		for (const message& m : message_buffer_) {
			if (m.id == T::id) {
				return true;
			}
		}

		return false;
	}

	bool has_error() const;
	void clear();
	void print_all() const;

	template <typename T>
	void print_only() const{
		for (const message& m : message_buffer_) {
			if (m.id == T::id) {
				if (m.error) {
					std::cerr << m.text << std::endl;
				} else {
					std::cout << m.text << std::endl;
				}
			}
		}
	}
private:
	template<typename T>
	void insert(const std::string& argument) {
		message newMessage;
		newMessage.id = T::id;
		newMessage.error = T::error;

		if (T::date) {
			time_t tt;
			tt = std::chrono::system_clock::to_time_t (std::chrono::system_clock::now());
			std::string date(ctime(&tt));
			date.resize(date.length() - 1);
			newMessage.text = "[" + date + "] " + T::text;
		} else {
			newMessage.text = T::text;
		}

		if (argument != "") {
			newMessage.text += " [" + argument + "]";
		}

		if (T::print) {
			if (T::error) {
				std::cerr << newMessage.text << std::endl;
			} else {
				std::cout << newMessage.text << std::endl;
			}
		}

		if (!T::hide) {
			message_buffer_.push_back(std::move(newMessage));
		}
	}

	struct message {
		int id;
		bool error;
		std::string text;
	};

	std::vector<message> message_buffer_;
	bool error_ = false;

};

#endif
