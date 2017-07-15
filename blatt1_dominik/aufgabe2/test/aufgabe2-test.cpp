#include "../../../deps/catch/include/catch.hpp"
#include "../src/barrier.hpp"
#include "../src/worker.hpp"
#include <thread>
#include <mutex>
#include <atomic>


TEST_CASE("Barrieren-Test", "[Barriere a]") {

    std::vector<std::thread> threads;
    std::vector<int> vec;
    std::atomic<int> counter {0};

    SECTION("Vier Threads warten an einer Barriere") {

        const int thread_count = 4;
        std::mutex mut;
        Barrier b(thread_count);

        for (int i = 0; i < thread_count; ++i) {
            threads.push_back(std::thread([&]() {
                ++counter;
                b.wait();
                int local_counter = counter.load();
                std::lock_guard<std::mutex> lock(mut);
                vec.push_back(local_counter);
            }));
        }

        for (int i = 0; i < thread_count; i++) {
            threads[i].join();
        }

        for (int i = 0; i < thread_count; i++) {
            REQUIRE(vec[i] == thread_count);
        }

    }

    SECTION("5'000 Threads warten an einer Barriere") {

        const int thread_count = 5000;
        std::mutex mut;
        Barrier b(thread_count);

        for (int i = 0; i < thread_count; ++i) {
            threads.push_back(std::thread([&]() {
                ++counter;
                b.wait();
                int local_counter = counter.load();
                std::lock_guard<std::mutex> lock(mut);
                vec.push_back(local_counter);
            }));
        }

        for (int i = 0; i < thread_count; i++) {
            threads[i].join();
        }

        for (int i = 0; i < thread_count; i++) {
            REQUIRE(vec[i] == thread_count);
        }
    }
}

TEST_CASE("Sequentielle Durchschnittsberechnung") {
    int num = 1;
    int id = 0;

    std::vector<double> elements {8, 7, 9, 10, 6};
    std::vector<double> average, variance;
    double global_avg = 0, global_var = 0;

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






