#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

#include "sudoku.h"
#include "solver.h"
#include "thread_pool.h"

int main(int argc, char** argv) {

    Thread_Pool<Solver> threadPool(31);

    Sudoku s("5  37   1"
             "4 1 8    "
             "       4 "
             " 542 7  8"
             "7       5"
             "  31 5 97"
             "93  5 81 "
             "         "
             "         ");

    s.print();

    locked_structure<std::vector<Sudoku>> results;

    Solver task(s, results);

    auto t1 = std::chrono::high_resolution_clock::now();

    threadPool.insert(task);

    threadPool.wait();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "time: " << duration << std::endl;

    for (size_t i = 0; i < results.obj.size(); i++) {
        results.obj[i].print();
    }


    return 0;
}
