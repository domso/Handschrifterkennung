#include <iostream>
#include <cuda.h>
#include <vector>
#include <chrono>
#include "neuronal_network.h"
#include "../sample_set.h"
#include "../sample.h"


/**
 * Train the network by processing the MNIST training set and updating the weights
 * @param nn the neuronal network to be trained
 * @param trainingSamples the training set
 */
void trainNetwork(neuronal_network& nn, std::vector<data::sample<float>>& trainingSamples){
	int errorCount = 0;
/*
	for(int i = 0; i < trainingSamples.size(); i++){
		std::vector<float>& input = trainingSamples[i].internal_data();
		int label = trainingSamples[i].get_label();

		int classification = nn.proccess_input(input, label, true, 8);
		if(classification != label){
			errorCount++;
		}
	}*/

	errorCount = nn.proccess_input(trainingSamples, true, 8);

	std::cout << "training completed!\n => " << errorCount << " mistakes out of " << trainingSamples.size() << " images (" << ((float)(trainingSamples.size() - errorCount) / trainingSamples.size() * 100) << "% sucess rate)\n";
}


/**
 * Test the network by processing the MNIST test set
 * @param nn  the neuronal network to be tested
 * @param testSamples  the test set
 */
void testNetwork(neuronal_network& nn, std::vector<data::sample<float>>& testSamples){
	int errorCount = 0;/*
	for(int i = 0; i < testSamples.size(); i++){
		std::vector<float>& input = testSamples[i].internal_data();
		int label = testSamples[i].get_label();


		if(classification != label){
			errorCount++;
		}
	}*/

	errorCount = nn.proccess_input(testSamples, false, 8);


	std::cout << "test completed!\n => " << errorCount << " mistakes out of " << testSamples.size() << " images (" << ((float)(testSamples.size() - errorCount) / testSamples.size() * 100) << "% sucess rate)\n";

}


int main() {
	std::vector<data::sample<float>> trainingInput = data::sample_set::load<float>("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte");

	int imgCount = trainingInput.size();
	if(imgCount == 0){
		std::cout << "no images were loaded, exiting....";
		return -1;
	}

	int inputCount  = trainingInput[0].size();
	int hiddenCount = 20;
	int outputCount = 10;

	neuronal_network nn(inputCount, hiddenCount, outputCount);

	layer& input = nn.get_layer(INPUT);
	layer& hidden = nn.get_layer(HIDDEN);
	layer& output = nn.get_layer(OUTPUT);

	std::cout << input.get_node_count() << " input nodes, " << hidden.get_node_count() << " hidden nodes, " << output.get_node_count() << " output nodes\n";
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
