/*
 * NeuronalNetwork.cpp
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#include "NeuronalNetwork.h"
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <thread>

/**
 *
 * @param inputCount  Number of nodes in the INPUT layer
 * @param hiddenCount Number of nodes in the HIDDEN layer
 * @param outputCount Number of nodes in the OUTPUT layer
 */
NeuronalNetwork::NeuronalNetwork(int inputCount, int hiddenCount, int outputCount) {
	learningRate = 0.2;
	layers.reserve(3);

	layers[0] = createInputLayer(inputCount);
	layers[1] = createLayer(hiddenCount, inputCount);
	layers[2] = createLayer(outputCount, hiddenCount);

	initWeights(HIDDEN);
	initWeights(OUTPUT);
}

NeuronalNetwork::~NeuronalNetwork() {
	for(Layer* l: layers){
		delete l;
	}
}

/**
 * Returns the corresponding layer to the given lType of the network
 * @param lType  Type of layer to be returned (INPUT, HIDDEN, OUTPUT)
 * @return the corresponding layer to lType
 */
Layer* NeuronalNetwork::getLayer(LayerType lType){
	switch(lType){
		case INPUT:
		{
			return layers[0];
		}
		case HIDDEN:
		{
			return layers[1];
		}
		case OUTPUT:
		{
			return layers[2];
		}
		default:
			return nullptr;
	}
}


/**
 * Creates an input layer
 * @param inputCount  number of nodes in the input-layer
 * @return
 */
Layer* NeuronalNetwork::createInputLayer(int inputCount) {
	return createLayer(inputCount, 0); // input layer has 0 weights!
}


/**
 * Creates a layer
 * @param nodeCount
 * @param weightCount
 * @return
 */
Layer* NeuronalNetwork::createLayer(int nodeCount, int weightCount) {
	Layer* layer = new Layer(nodeCount);

	for(int i = 0; i < nodeCount; i++){
		Node* node = new Node(0.0, 0.0, weightCount);
		layer->setNode(node, i);
	}

	return layer;
}


/**
 * Initializes a layer's weights with random values
 * @param lType  Defining what layer to initialize
 */
void NeuronalNetwork::initWeights(LayerType lType) {
	srand(time(NULL));
	Layer* layer = getLayer(lType);
	int nodeCount = layer->getNodeCount();

	for(int i = 0; i < nodeCount; i++){
		Node* node = layer->getNode(i);
		for(int j = 0; j < node->getWeights().size(); j++){
			node->getWeights()[j] = 0.7 * (rand() / (float) RAND_MAX);
			if(j%2)
				node->getWeights()[j] = - node->getWeights()[j];  // make half of the weights negative
		}
	}
}


/**
 * Returns the network's classification using the ID of the node with the highest output
 * @return
 */
int NeuronalNetwork::getNetworkClassification() {
	Layer* layer = getLayer(OUTPUT);
	Node* max = layer->getNode(0);

	for(int i = 1; i < layer->getNodeCount(); i++){
		Node* node = layer->getNode(i);
		if(node->getOutput() > max->getOutput())
			max = node;
	}

	return (max != nullptr) ? max->getOutput() : -1;
}


/**
 * Feeds some data into the input-layer of the network
 * @param input  a vector with the input values
 */
void NeuronalNetwork::feedInput(std::vector<float> input) {
	Layer* inputLayer = getLayer(INPUT);

	int threadCount = std::thread::hardware_concurrency();
	int elementsPerThread = input.size() / threadCount;
	std::vector<std::thread> threads(threadCount);

	for(int thID = 0; thID < threadCount; thID++){
		int rangeFrom = thID * elementsPerThread;
		int rangeTo = (thID == (threadCount - 1) ? input.size() : ((thID + 1) * elementsPerThread) );
		threads.push_back(std::thread([rangeFrom, rangeTo, &input, &inputLayer]{
			// copy the input values to the inputLayer
			for(int i = rangeFrom; i < rangeTo; i++){
				Node* inputNode = inputLayer->getNode(i);
				inputNode->setOutput(input[i]);
			}
		}));
	}

	for(int thID = 0; thID < threadCount; thID++){
		threads[thID].join();
	}

	// sequentiell
//	Layer* inputLayer = getLayer(INPUT);
//
//	// copy the input values to the inputLayer
//	for(int i = 0; i < input.size(); i++){
//		Node* inputNode = inputLayer->getNode(i);
//		inputNode->setOutput(input[i]);
//	}
}


/**
 * Feeds input layer values forward to the hidden- and then to the output-layer (calculation and activation fct)
 */
void NeuronalNetwork::feedForwardNetwork() {
	calcLayer(HIDDEN);
	calcLayer(OUTPUT);
}


/**
 * Back propagates network error from output-layer to hidden-layer
 * @param targetClassification  correct classification of the input stream (label)
 */
void NeuronalNetwork::backPropagateNetwork(int targetClassification) {
	backPropagateOutputLayer(targetClassification);
	backPropagateHiddenLayer(targetClassification);
}


/**
 * Updates a node's weights based on given delta
 * @param lType   Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param nodeID  id of the node to update
 * @param delta   difference between desired output and actual output
 */
void NeuronalNetwork::updateNodeWeights(LayerType lType, int nodeID, float delta){
	Layer* updateLayer = getLayer(lType);
	Node*  updateNode  = updateLayer->getNode(nodeID);

	Layer* prevLayer;
	if(lType == OUTPUT){
		prevLayer = getLayer(HIDDEN);
	}
	else{ // lType == HIDDEN
		prevLayer = getLayer(INPUT);
	}

	// go through all weights of updateNode and update them with the delta
	int i = 0;
	for(float weight: updateNode->getWeights()){
		Node* prevLayerNode = prevLayer->getNode(i);
		weight += (learningRate * prevLayerNode->getOutput() * delta);
		i++;
	}

	// update bias weigth
	updateNode->setBias(updateNode->getBias() + learningRate * delta);
}


/**
 * Performs the SEGMOID activiation function to the specified node
 * @param lType  Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param nodeID id of the node to activate
 */
void NeuronalNetwork::activateNode(LayerType lType, int nodeID) {
	Layer* layer = getLayer(lType);
	Node*  node  = layer->getNode(nodeID);

	node->setOutput( 1 / (1 + std::exp(-node->getOutput()))); // SIGMOID activation function
}


/**
 * Calculates the output value of the specified node by multiplying all its weights with the previous layer's outputs
 * @param lType   Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param nodeID  id of the node to calculate
 */
void NeuronalNetwork::calcNodeOutput(LayerType lType, int nodeID) {
	Layer* calcLayer = getLayer(lType);
	Node*  calcNode  = calcLayer->getNode(nodeID);

	Layer* prevLayer = getLayer(INPUT);
	if(lType == OUTPUT){
		prevLayer = getLayer(HIDDEN);
	}

	calcNode->setOutput(calcNode->getBias()); // start with the nodes bias
	for(int i = 0; i < prevLayer->getNodeCount(); i++){
		Node* prevLayerNode = prevLayer->getNode(i);
		calcNode->setOutput(calcNode->getOutput() + prevLayerNode->getOutput() * calcNode->getWeights()[i]);
	}
}


/**
 * Calculates the output values of the given layer
 * @param lType  Type of layer (INPUT, HIDDEN, OUTPUT)
 */
void NeuronalNetwork::calcLayer(LayerType lType) {
	Layer* layer = getLayer(lType);

	int threadCount = std::thread::hardware_concurrency();
	int nodesPerThread = layer->getNodeCount() / threadCount;
	std::vector<std::thread> threads(threadCount);

	for(int thID = 0; thID < threadCount; thID++){
		int rangeFrom = thID * nodesPerThread;
		int rangeTo = (thID == (threadCount - 1) ? layer->getNodeCount() : ((thID + 1) * nodesPerThread) );
		threads.push_back(std::thread([this, rangeFrom, rangeTo, lType]{

			for(int i = rangeFrom; i < rangeTo; i++){
				calcNodeOutput(lType, i);
				activateNode(lType, i);
			}
		}));
	}

	for(int thID = 0; thID < threadCount; thID++){
		threads[thID].join();
	}

	// seqentiell
//	for(int i = 0; i < layer->getNodeCount(); i++){
//		calcNodeOutput(lType, i);
//		activateNode(lType, i);
//	}
}


/**
 * Back propagates network error to hidden-layer
 * @param targetClassification  correct classification of the input stream (label)
 */
void NeuronalNetwork::backPropagateHiddenLayer(int targetClassification) {
	Layer* outputLayer = getLayer(OUTPUT);
	Layer* hiddenLayer = getLayer(HIDDEN);

	int threadCount = std::thread::hardware_concurrency();
	int nodesPerThread = hiddenLayer->getNodeCount() / threadCount;
	std::vector<std::thread> threads(threadCount);

	for(int thID = 0; thID < threadCount; thID++){
		int rangeFrom = thID * nodesPerThread;
		int rangeTo = (thID == (threadCount - 1) ? hiddenLayer->getNodeCount() : ((thID + 1) * nodesPerThread) );
		threads.push_back(std::thread([this, rangeFrom, rangeTo, &hiddenLayer, &outputLayer, targetClassification]{
			for(int i = rangeFrom; i < rangeTo; i++){
				Node* hiddenNode = hiddenLayer->getNode(i);
				float outputErrorSum = 0;

				for(int k = 0; k < outputLayer->getNodeCount(); k++){
					Node* outputNode = outputLayer->getNode(k);
					int targetOutput = (k == targetClassification) ? 1 : 0;

					float errorDelta  = targetOutput - outputNode->getOutput();
					float errorSignal = errorDelta *  outputNode->getOutput() * (1 - outputNode->getOutput());	// derivative of the SIGMOID activation function
					outputErrorSum += errorSignal * outputNode->getWeights()[i];
				}

				double hiddenErrorSignal = outputErrorSum * hiddenNode->getOutput() * (1 - hiddenNode->getOutput()); // derivative of the SIGMOID activation function
				updateNodeWeights(HIDDEN, i, hiddenErrorSignal);
			}
		}));
	}

	for(int thID = 0; thID < threadCount; thID++){
		threads[thID].join();
	}

	// sequentiell
//	for(int i = 0; i < hiddenLayer->getNodeCount(); i++){
//		Node* hiddenNode = hiddenLayer->getNode(i);
//		float outputErrorSum = 0;
//
//		for(int k = 0; k < outputLayer->getNodeCount(); k++){
//			Node* outputNode = outputLayer->getNode(k);
//			int targetOutput = (k == targetClassification) ? 1 : 0;
//
//			float errorDelta  = targetOutput - outputNode->getOutput();
//			float errorSignal = errorDelta *  outputNode->getOutput() * (1 - outputNode->getOutput());	// derivative of the SIGMOID activation function
//			outputErrorSum += errorSignal * outputNode->getWeights()[i];
//		}
//
//		double hiddenErrorSignal = outputErrorSum * hiddenNode->getOutput() * (1 - hiddenNode->getOutput()); // derivative of the SIGMOID activation function
//		updateNodeWeights(HIDDEN, i, hiddenErrorSignal);
//	}
}


/**
 * Back propagates network error to output-layer
 * @param targetClassification  correct classification of the input stream (label)
 */
void NeuronalNetwork::backPropagateOutputLayer(int targetClassification) {
	Layer* outputLayer = getLayer(OUTPUT);

	int threadCount = std::thread::hardware_concurrency();
	int nodesPerThread = outputLayer->getNodeCount() / threadCount;
	std::vector<std::thread> threads(threadCount);

	for(int thID = 0; thID < threadCount; thID++){
		int rangeFrom = thID * nodesPerThread;
		int rangeTo = (thID == (threadCount - 1) ? outputLayer->getNodeCount() : ((thID + 1) * nodesPerThread) );
		threads.push_back(std::thread([this, rangeFrom, rangeTo, &outputLayer, targetClassification]{
			for(int i = rangeFrom; i < rangeTo; i++){
				Node* outputNode = outputLayer->getNode(i);
				int targetOutput = (i == targetClassification) ? 1 : 0;

				float errorDelta  = targetOutput - outputNode->getOutput();
				float errorSignal = errorDelta *  outputNode->getOutput() * (1 - outputNode->getOutput());	// derivative of the SIGMOID activation function

				updateNodeWeights(OUTPUT, i, errorSignal);
			}
		}));
	}

	for(int thID = 0; thID < threadCount; thID++){
		threads[thID].join();
	}

	// sequentiell
//	for(int i = 0; i < outputLayer->getNodeCount(); i++){
//		Node* outputNode = outputLayer->getNode(i);
//		int targetOutput = (i == targetClassification) ? 1 : 0;
//
//		float errorDelta  = targetOutput - outputNode->getOutput();
//		float errorSignal = errorDelta *  outputNode->getOutput() * (1 - outputNode->getOutput());	// derivative of the SIGMOID activation function
//
//		updateNodeWeights(OUTPUT, i, errorSignal);
//	}
}
