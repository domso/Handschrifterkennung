#include <iostream>
#include "../data/sample_set.h"
#include "cuda_neuronal_network.h"
#include "cuda_model.h"
#include "../gui/basic_interface.h"
#include <thread>
#include <chrono>
#include "../util/config_file.h"
#include "../parameters.h"

namespace cuda {

void training(cuda::model& model, cuda::neuronal_network& NN, std::vector<data::sample<float>>& trainingsData, const int numRelearning) {
	auto tp1 = std::chrono::high_resolution_clock::now();
	NN.train(model, trainingsData, numRelearning);
	auto tp2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast < std::chrono::microseconds > (tp2 - tp1).count();
	std::cout << "Training took: " << duration / (double) 1000000 << "sec" << std::endl;
}

void testing(cuda::model& model, cuda::neuronal_network& NN, std::vector<data::sample<float>>& testData) {
	auto tp1 = std::chrono::high_resolution_clock::now();
	auto result = NN.test(model, testData);
	auto tp2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast < std::chrono::microseconds > (tp2 - tp1).count();
	std::cout << "Testing took: " << duration / (double) 1000000 << "sec" << std::endl;

	std::cout << result.correct << std::endl;
	std::cout << result.total << std::endl;
	std::cout << result.ratio << std::endl;
}

bool main(cuda::neuronal_network& NN, std::vector<data::sample<float>>& trainingsData, std::vector<data::sample<float>>& testData, const int useGui, util::config_file& config) {
	auto numHidden = config.getNumeric<int, parameters::num_hidden>();
	auto numRelearning = config.getNumeric<int, parameters::num_relearning>();
	auto learningRate = config.getNumeric<float, parameters::learning_rate>();

	cuda::model model;
	cuda::neuronal_network::config_t configNN;

	configNN.numHidden = numHidden;
	configNN.learningRate = learningRate;

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
