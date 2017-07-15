#include <iostream>
#include <vector>
#include <iterator>
#include <thread>
#include <mutex>
#include "counter.hpp"

template <typename T>
void incTest(T* c, int n) {
    for (int i = 0; i < n; i++) {
        (*c)++;
    }
}

int main(int argc, char** argv) {

    Counter c1;
    int c2 = 0;
    int n = 1000000;


    std::thread t1(&incTest<Counter>, &c1, n);
    std::thread t2(&incTest<Counter>, &c1, n);
    std::thread t3(&incTest<Counter>, &c1, n);
    std::thread t4(&incTest<Counter>, &c1, n);

    std::thread t5(&incTest<int>, &c2, n);
    std::thread t6(&incTest<int>, &c2, n);
    std::thread t7(&incTest<int>, &c2, n);
    std::thread t8(&incTest<int>, &c2, n);


    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    t7.join();
    t8.join();

    std::cout << "Sollwert: " << n * 4 << std::endl;
    std::cout << "Istwert(safe): " << c1.get() << std::endl;
    std::cout << "Istwert(unsafe): " << c2 << std::endl;

    return 0;
}
