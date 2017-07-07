#include <iostream>
#include <cuda.h>
#include <vector>
#include "NeuronalNetwork.h"
#include "sample_set.h"
#include "sample.h"


/**
 * Training the network by processing the MNIST training set and updating the weights
 * @param nn
 */
void trainNetwork(NeuronalNetwork& nn, std::vector<data::sample<float>>& trainingSamples){
	int errorCount = 0;

	for(int i = 0; i < trainingSamples.size(); i++){
		std::vector<float>& input = trainingSamples[i].internalData();
		int label = trainingSamples[i].getLabel();

		nn.feedInput(input);

		nn.feedForwardNetwork();

		nn.backPropagateNetwork(label);

		int classification = nn.getNetworkClassification();
		if(classification != label){
//			std::cout << "network computed " << classification << ", but label is " << label << "\n";
			errorCount++;
		}
//		else
//		{
//			std::cout << "network computed " << classification << " and label is " << label << "\n";
//		}
	}
	std::cout << "training completed!\n => " << errorCount << " mistakes out of " << trainingSamples.size() << " images\n";
}

void testNetwork(NeuronalNetwork& nn, std::vector<data::sample<float>>& testSamples){
	int errorCount = 0;
	for(int i = 0; i < testSamples.size(); i++){
		std::vector<float>& input = testSamples[i].internalData();
		int label = testSamples[i].getLabel();

		nn.feedInput(input);

		nn.feedForwardNetwork();

		int classification = nn.getNetworkClassification();
		if(classification != label){
			errorCount++;
		}
	}
	std::cout << "test completed!\n => " << errorCount << " mistakes out of " << testSamples.size() << " images\n";

}

int main(int argc, char** argv) {
	std::vector<data::sample<float>> trainingInput = data::sample_set::load<float>("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte");

	int imgCount    = trainingInput.size();
	if(imgCount == 0){
		std::cout << "no images were loaded, exiting....";
		return -1;
	}

	int inputCount  = trainingInput[0].size();
	int hiddenCount = 20;
	int outputCount = 10;

	NeuronalNetwork nn(inputCount, hiddenCount, outputCount);

	Layer* input = nn.getLayer(INPUT);
	Layer* hidden = nn.getLayer(HIDDEN);
	Layer* output = nn.getLayer(OUTPUT);

	std::cout << input->getNodeCount() << " input nodes, " << hidden->getNodeCount() << " hidden nodes, " << output->getNodeCount() << " output nodes\n";
	std::vector<data::sample<float>> testInput = data::sample_set::load<float>("./t10k-images.idx3-ubyte", "./t10k-labels.idx1-ubyte");

for(int i = 0; i < 1; i++){

	trainNetwork(nn, trainingInput);
	testNetwork(nn, testInput);
}


}
