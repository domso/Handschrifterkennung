#ifndef util_conv_endian_h
#define util_conv_endian_h

#include <stdint.h>

namespace util {
/**
 * Converts a Big-Endian to a Little-Endian and vice versa
 * @param input: original instance of T
 * @return: converted T
 */
template<typename T>
T conv_endian(const T input) {
	T result;

	for (int i = 0; i < sizeof(T); i++) {
		((int8_t*) (&result))[sizeof(T) - i - 1] = ((int8_t*) (&input))[i];
	}

	return result;
}
}

#endif
