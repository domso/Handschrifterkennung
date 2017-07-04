#ifndef helper_h
#define helper_h

#include <stdint.h>

namespace helper {
    template <typename T>
    T convEndian(const T input) {
        T result;
        
        for (int i = 0; i < sizeof(T); i++) {
            ((int8_t*)(&result))[sizeof(T) - i - 1] = ((int8_t*)(&input))[i];
        }
     
        return result;
    }
}

#endif
