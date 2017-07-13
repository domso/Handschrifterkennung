#include <iostream>
#include <cuda.h>
#include <vector>
#include <chrono>
#include "NeuronalNetwork.h"
#include "../sample_set.h"
#include "../sample.h"


/**
 * Training the network by processing the MNIST training set and updating the weights
 * @param nn
 */
void trainNetwork(NeuronalNetwork& nn, std::vector<data::sample<float>>& trainingSamples){
	int errorCount = 0;

	for(int i = 0; i < trainingSamples.size(); i++){
		std::vector<float>& input = trainingSamples[i].internal_data();
		int label = trainingSamples[i].get_label();

		nn.feedInput(input);

		nn.feedForwardNetwork();

		nn.backPropagateNetwork(label);

		int classification = nn.getNetworkClassification();
		if(classification != label){
			errorCount++;
		}
	}
	std::cout << "training completed!\n => " << errorCount << " mistakes out of " << trainingSamples.size() << " images (" << ((float)(trainingSamples.size() - errorCount) / trainingSamples.size() * 100) << "% sucess rate)\n";
}

void testNetwork(NeuronalNetwork& nn, std::vector<data::sample<float>>& testSamples){
	int errorCount = 0;
	for(int i = 0; i < testSamples.size(); i++){
		std::vector<float>& input = testSamples[i].internal_data();
		int label = testSamples[i].get_label();

		nn.feedInput(input);

		nn.feedForwardNetwork();

		int classification = nn.getNetworkClassification();
		if(classification != label){
			errorCount++;
		}
	}
	std::cout << "test completed!\n => " << errorCount << " mistakes out of " << testSamples.size() << " images (" << ((float)(testSamples.size() - errorCount) / testSamples.size() * 100) << "% sucess rate)\n";

}

int main2() {
	std::vector<data::sample<float>> trainingInput = data::sample_set::load<float>("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte");

	int imgCount    = trainingInput.size();
	if(imgCount == 0){
		std::cout << "no images were loaded, exiting....";
		return -1;
	}

	int inputCount  = trainingInput[0].size();
	int hiddenCount = 50;
	int outputCount = 10;

	NeuronalNetwork nn(inputCount, hiddenCount, outputCount);

	Layer* input = nn.getLayer(INPUT);
	Layer* hidden = nn.getLayer(HIDDEN);
	Layer* output = nn.getLayer(OUTPUT);

	std::cout << input->getNodeCount() << " input nodes, " << hidden->getNodeCount() << " hidden nodes, " << output->getNodeCount() << " output nodes\n";
	std::vector<data::sample<float>> testInput = data::sample_set::load<float>("./t10k-images.idx3-ubyte", "./t10k-labels.idx1-ubyte");

	auto startTimeTrain = std::chrono::high_resolution_clock::now();
	trainNetwork(nn, trainingInput);
	auto endTimeTrain   = std::chrono::high_resolution_clock::now();
	auto elapsedTimeTrain = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeTrain - startTimeTrain).count();
	std::cout << "\nTraining with " << trainingInput.size() << " pictures ended after " << elapsedTimeTrain << "ms\n\n";

	auto startTimeTest = std::chrono::high_resolution_clock::now();
	testNetwork(nn, testInput);
	auto endTimeTest   = std::chrono::high_resolution_clock::now();
	auto elapsedTimeTest = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeTest - startTimeTest).count();
	std::cout << "\nTesting with " << testInput.size() << " pictures ended after " << elapsedTimeTest << "ms\n\n";

	return 0;
}
