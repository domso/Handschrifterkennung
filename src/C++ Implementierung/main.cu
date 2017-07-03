#include <iostream>
#include <cuda.h>

void trainNetwork(NeuronalNetwork* nn){
	//TODO
}

int main(int argc, char** argv) {
	int inputCount = 0;
	int hiddenCount = 20;
	int outputCount = 10;

	NeuronalNetwork* nn = new NeuronalNetwork(inputCount, hiddenCount, outputCount);


	delete nn;
}
