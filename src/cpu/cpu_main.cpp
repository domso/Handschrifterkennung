#include <iostream>
#include <vector>
#include <chrono>
#include "neuronal_network.h"
#include "../data/sample_set.h"
#include "../data/sample.h"
#include "../util/config_file.h"
#include "../parameters.h"

namespace cpu {

	/**
	 * trains the given network with the given data
	 * @param NN  the network to be trained
	 * @param trainingsData  the data to use for the training
	 * @param numRelearning  the number of iterations through the training data
	 * @param numThreads   the number threads to use
	 */
	void training(cpu::neuronal_network& NN, const std::vector<data::sample<float>>& trainingsData, const int numRelearning, const int numThreads) {
		for(int i = 0; i < numRelearning; i++){
			auto tp1 = std::chrono::high_resolution_clock::now();
			NN.proccess_input(trainingsData, true, numThreads);
			auto tp2 = std::chrono::high_resolution_clock::now();

			auto duration = std::chrono::duration_cast < std::chrono::microseconds > (tp2 - tp1).count();
			std::cout << "Training took: " << duration / (double) 1000000 << "sec" << std::endl;
		}
	}

	/**
	 * tests the given network with the given data
	 * @param NN  the network to be trained
	 * @param testData  the data to use for the training
	 * @param numThreads   the number threads to use
	 */
	void testing(cpu::neuronal_network& NN, const std::vector<data::sample<float>>& testData, const int numThreads) {
		auto tp1 = std::chrono::high_resolution_clock::now();
		auto result = NN.proccess_input(testData, false, numThreads);
		auto tp2 = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast < std::chrono::microseconds > (tp2 - tp1).count();
		std::cout << "Testing took: " << duration / (double) 1000000 << "sec" << std::endl;

		std::cout << "Correct: " << testData.size() - result << std::endl;
		std::cout << "Total: " << testData.size() << std::endl;
		std::cout << "Ratio: " << (testData.size() - result) / (double) testData.size() << std::endl;
	}

	bool main(cpu::neuronal_network& NN, const std::vector<data::sample<float>>& trainingsData, const std::vector<data::sample<float>>& testData, const int useGui, const util::config_file& config) {
		auto numRelearning = config.getNumeric<int, parameters::num_relearning>();
		auto learningRate = config.getNumeric<float, parameters::learning_rate>();
		auto numThreads = config.getNumeric<int, parameters::num_threads>();

		NN.set_learning_rate(learningRate);

		std::cout << "C++ implementation:" << std::endl;
		training(NN, trainingsData, numRelearning, numThreads);
		testing(NN, testData, numThreads);

		return true;
	}
}
