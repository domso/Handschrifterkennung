#ifndef util_logger_h
#define util_logger_h

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <ostream>

namespace util {
/*
 * Class for simple logs and error-handling:
 * The possible log-messages are predefined in separate classes.
 * Example:
 *
 *	struct example_error {
 *		constexpr static int id = 0;             // ID for identification with logger::check() and logger::print_only()
 *		constexpr static bool critical = false;  // abort execution in logger::log()
 *		constexpr static bool error = true; 	 // just a warning / info?
 *		constexpr static bool date = false;		 // add the current date
 *		constexpr static bool print = true;	     // print every logger::log() call to std::cout
 *		constexpr static bool hide = false;		 // do not store the message in the logger-buffer
 *		constexpr static const auto text = "\o/";// output message-text
 *	};
 *
 *	void test() {
 *		logger l;
 *
 *		l.log<example_error>("optional argument");
 *		// output in std::cerr: \o/ [optional argument]
 *
 *		if (l.has_error()) { // true, because of runtime_error::error == true
 *			if (l.check<example_error>()) { // true, because of runtime_error::hide == false
 *				l.print_only<example_error>(); // not very efficient here
 *				// output in std::cerr: \o/ [optional argument]
 *			}
 *		}
 *
 *		l.clear(); // clear all messages
 *	}
 */
class logger {
public:
	/*
	 * Logs a new occurrence of T with the given argument
	 * T requires all fields used in the general example
	 * @param argument: optional argument for the event (displayed as [argument])
	 */
	template<typename T>
	void log(const std::string argument = "") {
		insert<T>(argument);

		if (T::error) {
			m_error = true;
		}

		if (T::critical) {
			print_all();

			std::cerr << "Execution reached critical error!" << std::endl;
			std::abort();
		}
	}

	/*
	 * Searches for an occurrence of T
	 * T requires all fields used in the general example
	 * T is identified by T::id!
	 * @return: true, if T was logged at least once
	 */
	template<typename T>
	bool check() const {
		for (const message& m : m_messageBuffer) {
			if (m.id == T::id) {
				return true;
			}
		}

		return false;
	}

	/*
	 * Searches and prints all occurrences of T to std::cout
	 * or to std::cerr, if T::error is marked as true
	 * T requires all fields used in the general example
	 * T is identified by T::id!
	 */
	template<typename T>
	void print_only() const {
		for (const message& m : m_messageBuffer) {
			if (m.id == T::id) {
				if (T::error) {
					std::cerr << m.text << std::endl;
				} else {
					std::cout << m.text << std::endl;
				}
			}
		}
	}

	/*
	 * Prints all messages to std::cout
	 * or to std::cerr, if message was marked as an error
	 */
	void print_all() const;

	/*
	 * This does not scan the internal-data --> very fast!
	 * @return: true, if the number of errors equals zero
	 */
	bool has_error() const;

	/*
	 * Deletes all messages and resets the logger
	 */
	void clear();
private:

	/*
	 * Inserts T with the given argument to the internal data-structure
	 * @param argument: (see logger::log)
	 */
	template<typename T>
	void insert(const std::string& argument) {
		message newMessage;
		newMessage.id = T::id;
		newMessage.error = T::error;

		if (T::date) {
			time_t tt;
			tt = std::chrono::system_clock::to_time_t(
					std::chrono::system_clock::now());
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
			m_messageBuffer.push_back(std::move(newMessage));
		}
	}

	/*
	 * Internal data-structure for a message
	 */
	struct message {
		int id;
		bool error;
		std::string text;
	};

	std::vector<message> m_messageBuffer;
	bool m_error = false;

};
}
#endif
