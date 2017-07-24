#include <iostream>
#include <thread>
#include <chrono>
#include "../data/sample_set.h"
#include "cuda_neuronal_network.h"
#include "cuda_model.h"
#include "../gui/basic_interface.h"
#include "../util/config_file.h"
#include "../parameters.h"

namespace cuda {

/**
 * Trains the given model on the given neuronal-network with the given trainingsData
 * Prints the duration in microseconds
 * @param model: initialized model
 * @param NN: initialized neuronal-network
 * @param trainingsData: non empty vector containing the labeled trainings-samples
 * @param numRelearning: number of relearning iterations
 */
void training(cuda::model& model, const cuda::neuronal_network& NN, const std::vector<data::sample<float>>& trainingsData, const int numRelearning) {
	auto tp1 = std::chrono::high_resolution_clock::now();
	NN.train(model, trainingsData, numRelearning);
	auto tp2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast < std::chrono::microseconds > (tp2 - tp1).count();
	std::cout << "Training took: " << duration / (double) 1000000 << "sec" << std::endl;
}

/**
 * Tests the given model on the given neuronal-network with the given testData
 * Prints the duration in microseconds
 * @param model: trained model
 * @param NN: initialized neuronal-network
 * @param testData: non empty vector containing the labeled test-samples
 */
void testing(const cuda::model& model, const cuda::neuronal_network& NN, const std::vector<data::sample<float>>& testData) {
	auto tp1 = std::chrono::high_resolution_clock::now();
	auto result = NN.test(model, testData);
	auto tp2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast < std::chrono::microseconds > (tp2 - tp1).count();
	std::cout << "Testing took: " << duration / (double) 1000000 << "sec" << std::endl;

	std::cout << "Correct: " << result.correct << std::endl;
	std::cout << "Total: " << result.total << std::endl;
	std::cout << "Ratio: " << result.ratio  * 100 << "%" << std::endl;
}

bool main(cuda::neuronal_network& NN, const std::vector<data::sample<float>>& trainingsData, const std::vector<data::sample<float>>& testData, const int useGui, const util::config_file& config) {
	auto numHidden = config.getNumeric<int, parameters::num_hidden>();
	auto numRelearning = config.getNumeric<int, parameters::num_relearning>();
	auto learningRate = config.getNumeric<float, parameters::learning_rate>();

	std::cout << "Cuda implementation with " << numHidden << " hidden nodes:" << std::endl;
	cuda::model model;
	cuda::neuronal_network::config_t configNN;

	configNN.numOutput = 10;
	configNN.numHidden = numHidden;
	configNN.learningRate = learningRate;

	NN.set_config(configNN);

	training(model, NN, trainingsData, numRelearning);
	testing(model, NN, testData);

	if (useGui) {
		if (!NN.set_classify_context(model, trainingsData[0])) {
			std::cout << "Could not create context" << std::endl;
			return false;
		}
	}

	return true;
}

}
