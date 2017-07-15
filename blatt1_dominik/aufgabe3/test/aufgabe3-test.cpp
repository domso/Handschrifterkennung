#include "../../../deps/catch/include/catch.hpp"


#include "../src/sudoku.h"
#include "../src/solver.h"
#include "../src/locked_structure.h"
#include "../src/thread_pool.h"

#include <thread>
#include <mutex>
#include <atomic>

TEST_CASE("Sudoku-Test") {

    Sudoku s;
    REQUIRE(s.is_complete() == false);
    REQUIRE(s.is_valid() == false);


    std::string input("596374281"
                      "421689573"
                      "378512946"
                      "654297138"
                      "719863425"
                      "283145697"
                      "937456812"
                      "145728369"
                      "862931754");

    for (size_t pos = 0; pos < input.length(); pos++) {
        REQUIRE(s.tryInsert(pos, input[pos] - 48) == true);
    }

    REQUIRE(s.is_complete() == true);
    REQUIRE(s.is_valid() == true);

    s.set(27, 0);

    REQUIRE(s.is_complete() == false);
    REQUIRE(s.is_valid() == false);

    REQUIRE(s.getNextPos() == 27);


    Sudoku s2("596374281"
              "421689573"
              "378512946"
              "654297138"
              "719863425"
              "283145697"
              "937456812"
              "145728369"
              "862931754");


    REQUIRE(s2.is_complete() == true);
    REQUIRE(s2.is_valid() == true);
}

TEST_CASE("Thread-Pool-Test") {
    struct test_callable_t {
        void operator()(Thread_Pool<test_callable_t>* tp) {
            (*a)++;
        }
        std::atomic<int>* a;
    };

    Thread_Pool<test_callable_t> pool1(10);

    pool1.wait();
    pool1.close();

    Thread_Pool<test_callable_t> pool2(10);

    int result = 18648;
    std::atomic<int> current(0);

    for (int i = 0; i < result; i++) {

        test_callable_t task;
        task.a = &current;

        pool2.insert(task);
    }

    pool2.wait();

    REQUIRE(result == result);
}

TEST_CASE("Sudoku-Solver-Test") {

    Thread_Pool<Solver> pool1(10);
    locked_structure<std::vector<Sudoku>> results;
    Sudoku original("596374281"
                    "421689573"
                    "378512946"
                    "654297138"
                    "719863425"
                    "283145697"
                    "937456812"
                    "145728369"
                    "862931754");


    for (int i = 0; i < 81; i++) {
        Sudoku s("596374281"
                 "421689573"
                 "378512946"
                 "654297138"
                 "719863425"
                 "283145697"
                 "937456812"
                 "145728369"
                 "862931754");

        s.set(i, 0);
        Solver task(s, results);

        pool1.insert(task);

    }

    pool1.wait();

    for (Sudoku& s : results.obj) {
        REQUIRE(s == original);
    }

    results.obj.clear();

    {
        Sudoku s("596374281"
                 "421689573"
                 "378512946"
                 "654297138"
                 "719863425");

        Solver task(s, results);

        pool1.insert(task);
    }
    int numHit = 0;

    pool1.wait();

    for (Sudoku& s : results.obj) {

        if (s == original) {
            numHit++;
        }

        REQUIRE(s.is_valid() == true);
        REQUIRE(s.is_complete() == true);
    }

    REQUIRE(numHit > 0);

}


/*
TEST_CASE("Barrieren-Test", "[Barriere a]") {

    std::vector<std::thread> threads;
    std::vector<int> vec;
    std::atomic<int> counter{0};

    SECTION("Vier Threads warten an einer Barriere") {

    const int thread_count = 4;
    std::mutex mut;
    Barrier b(thread_count);

    for (int i=0; i<thread_count; ++i) {
        threads.push_back(std::thread([&](){
            ++counter;
            b.wait();
            int local_counter = counter.load();
            std::lock_guard<std::mutex> lock(mut);
            vec.push_back(local_counter);
            }));
    }

    for (int i=0; i<thread_count; i++) {
        threads[i].join();
    }

    for (int i=0; i<thread_count; i++) {
        REQUIRE(vec[i]== thread_count);
    }

    }

    SECTION("5'000 Threads warten an einer Barriere"){

    const int thread_count = 5000;
    std::mutex mut;
    Barrier b(thread_count);

    for (int i=0; i<thread_count; ++i) {
        threads.push_back(std::thread([&](){
            ++counter;
            b.wait();
            int local_counter = counter.load();
            std::lock_guard<std::mutex> lock(mut);
            vec.push_back(local_counter);
            }));
    }

    for (int i=0; i<thread_count; i++) {
        threads[i].join();
    }

    for (int i=0; i<thread_count; i++) {
        REQUIRE(vec[i]== thread_count);
    }
    }
}

TEST_CASE("Sequentielle Durchschnittsberechnung"){
     int num = 1;
     int id = 0;

     std::vector<double> elements {8,7,9,10,6};
     std::vector<double> average, variance;
     double global_avg=0, global_var=0;

     average.resize(num, 0);
     variance.resize(num, 0);
    Barrier b(1);
     Worker w(num, id, elements, average, variance, global_avg, global_var, b);

     w();

     REQUIRE(((int) w.local_average()) == 8);

     REQUIRE(((int) global_avg) == 8);
     REQUIRE(((int) w.local_variance(8)) == 2);

     REQUIRE(((int) global_var) == 2);
    }

*/
