#include <stdlib.h>

#include <thread>
#include <vector>
#include "worker.hpp"

int main(int argc, char** argv) {


    std::vector<double> data;

    data.push_back(1);
    data.push_back(8);
    data.push_back(7);
    data.push_back(8);
    data.push_back(7);
    data.push_back(2);
    data.push_back(3);
    data.push_back(2);
    data.push_back(9);
    data.push_back(8);

    std::vector<double> avgResults(2);
    std::vector<double> varResults(2);

    double avgResult;
    double varResult;

    Barrier b(2);

    Worker w1(2, 0, data, avgResults, varResults, avgResult, varResult, b);
    Worker w2(2, 1, data, avgResults, varResults, avgResult, varResult, b);

    std::thread t1(w1);
    std::thread t2(w2);

    t1.join();
    t2.join();

    std::cout << varResult << std::endl;
    std::cout << avgResult << std::endl;

    return 0;
}
