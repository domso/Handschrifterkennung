#include <iostream>
#include <vector>
#include <chrono>
#include "neuronal_network.h"
#include "../data/sample_set.h"
#include "../data/sample.h"
#include "../util/config_file.h"
#include "../parameters.h"

namespace cpu {

void training(cpu::neuronal_network& NN, std::vector<data::sample<float>>& trainingsData, const int numRelearning, const int numThreads) {
	auto tp1 = std::chrono::high_resolution_clock::now();
	NN.proccess_input(trainingsData, true, numThreads);
	auto tp2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast < std::chrono::microseconds > (tp2 - tp1).count();
	std::cout << "Training took: " << duration / (double) 1000000 << "sec" << std::endl;
}

void testing(cpu::neuronal_network& NN, std::vector<data::sample<float>>& testData, const int numThreads) {
	auto tp1 = std::chrono::high_resolution_clock::now();
	auto result = NN.proccess_input(testData, false, numThreads);
	auto tp2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast < std::chrono::microseconds > (tp2 - tp1).count();
	std::cout << "Testing took: " << duration / (double) 1000000 << "sec" << std::endl;

	std::cout << testData.size() - result << std::endl;
	std::cout << testData.size() << std::endl;
	std::cout << (testData.size() - result) / (double) testData.size() << std::endl;
}

bool main(cpu::neuronal_network& NN, std::vector<data::sample<float>>& trainingsData, std::vector<data::sample<float>>& testData, const int useGui, util::config_file& config) {
	auto numRelearning = config.getNumeric<int, parameters::num_relearning>();
	auto learningRate = config.getNumeric<float, parameters::learning_rate>();
	auto numThreads = config.getNumeric<int, parameters::num_threads>();

	training(NN, trainingsData, numRelearning, numThreads);
	testing(NN, testData, numThreads);

	return true;
}

}
