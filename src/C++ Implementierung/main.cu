#include <iostream>
#include <cuda.h>
#include <vector>
#include "NeuronalNetwork.h"
#include "util/mnist-utils.h"
#include "util/mnist-stats.h"

std::vector<float> getVectorFromImage(MNIST_Image *img){

	std::vector<float> v(MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT);

	for (int i = 0; i < v.size(); i++)
		v[i] = img->pixel[i] ? 1.0 : 0.0;
	return v;
}


/**
 * Training the network by processing the MNIST training set and updating the weights
 * @param nn
 */
void trainNetwork(NeuronalNetwork* nn){
//	FILE *imageFile, *labelFile;
//	imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
//	labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);
//	int errorCount = 0;
//
//	for(int i = 0; i < MNIST_MAX_TRAINING_IMAGES; i++){
//		MNIST_Image img;//   = getImage(imageFile);
//		MNIST_Label label;// = getLabel(labelFile);
//
//		std::vector<float> input = getVectorFromImage(&img);
//
//		nn->feedInput(input);
//
//		nn->feedForwardNetwork();
//
//		nn->backPropagateNetwork(label);
//
//		if(nn->getNetworkClassification() != label)
//			errorCount++;
//	}

//	fclose(imageFile);
//	fclose(labelFile);
}

int main(int argc, char** argv) {
	int inputCount = 0;
	int hiddenCount = 20;
	int outputCount = 10;

	NeuronalNetwork* nn = new NeuronalNetwork(inputCount, hiddenCount, outputCount);

	delete nn;
}
