#include "../../../deps/catch/include/catch.hpp"
#include "../src/counter.hpp"
#include <thread>

TEST_CASE("Counter kann sequentiell inkrementiert werden", "[Counter sequential]") {
    Counter c;

    REQUIRE(c.get() == 0);

    SECTION("Einmaliges Post-Inkrement erhöht den Zähler um 1") {
        c++;

        REQUIRE(c.get() == 1);
    }

    SECTION("49-faches Post-Inkrement erhöht den Zähler um 49") {
        const int upper_bound = 49;

        for (int i = 0; i < upper_bound; ++i) {
            c++;
        }

        REQUIRE(c.get() == upper_bound);
    }
}


TEST_CASE("Counter kann parallel inkrementiert werden", "[Counter parallel]") {
    Counter c;

    REQUIRE(c.get() == 0);

    std::vector<std::thread> threads;

    SECTION("Paralleles 10000-faches Post-Inkrement mit vier Threads erhöht den Zähler um 40'000") {

        const int thread_count = 4;
        const int local_count = 10000;

        for (int i = 0; i < thread_count; ++i) {
            threads.push_back(std::thread([&c]() {
                for (int i = 0; i < local_count; i++) {
                    c++;
                }
            }));
        }

        for (int i = 0; i < thread_count; i++) {
            threads[i].join();
        }

        REQUIRE(c.get() == (local_count * thread_count));
    }

    SECTION("Paralleles 10000-faches Post-Inkrement mit 16 Threads erhöht den Zähler um 160'000") {

        const int thread_count = 16;
        const int local_count = 10000;

        for (int i = 0; i < thread_count; ++i) {
            threads.push_back(std::thread([&c]() {
                for (int i = 0; i < local_count; i++) {
                    c++;
                }
            }));
        }

        for (int i = 0; i < thread_count; i++) {
            threads[i].join();
        }

        REQUIRE(c.get() == (local_count * thread_count));
    }

}
