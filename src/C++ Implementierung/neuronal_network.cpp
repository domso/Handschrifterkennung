/*
 * neuronal_network.cpp
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#include "neuronal_network.h"
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <iostream>

/**
 * @param inputCount  Number of nodes in the INPUT layer
 * @param hiddenCount Number of nodes in the HIDDEN layer
 * @param outputCount Number of nodes in the OUTPUT layer
 */
neuronal_network::neuronal_network(int inputCount, int hiddenCount, int outputCount) : m_layers(3) {
	srand(time(NULL));
	m_learning_rate = 0.2;

	m_layers[0] = create_input_layer(inputCount);
	m_layers[1] = create_layer(hiddenCount, inputCount);
	m_layers[2] = create_layer(outputCount, hiddenCount);

	init_weights(HIDDEN);
	init_weights(OUTPUT);
}

neuronal_network::~neuronal_network() {
	for(layer* l: m_layers){
		delete l;
	}
}

/**
 * Returns the corresponding layer to the given lType of the network
 * @param lType  Type of layer to be returned (INPUT, HIDDEN, OUTPUT)
 * @return the corresponding layer to lType
 */
layer* neuronal_network::get_layer(layer_type lType) const{
	switch(lType){
		case INPUT:
		{
			return m_layers[0];
		}
		case HIDDEN:
		{
			return m_layers[1];
		}
		case OUTPUT:
		{
			return m_layers[2];
		}
		default:
			return nullptr;
	}
}


/**
 * Creates an input layer
 * @param inputCount  number of nodes in the input-layer
 * @return the created layer
 */
layer* neuronal_network::create_input_layer(int inputCount) {
	return create_layer(inputCount, 0); // input layer has 0 weights!
}


/**
 * Creates a layer
 * @param nodeCount   the number of nodes that this layer should contain
 * @param weightCount the number of weights that should every node contain
 * @return the created layer
 */
layer* neuronal_network::create_layer(int nodeCount, int weightCount) {
	layer* l = new layer(nodeCount);

	for(int i = 0; i < nodeCount; i++){
		node* n = new node(0.0, 0.0, weightCount);
		l->set_node(n, i);
	}

	return l;
}


/**
 * Initializes a layer's weights with random values
 * @param lType  Defining what layer to initialize
 */
void neuronal_network::init_weights(layer_type lType) {
	layer* layer = get_layer(lType);
	int nodeCount = layer->get_node_count();

	for(int i = 0; i < nodeCount; i++){
		node* node = layer->get_node(i);

		int j = 0;
		for (float& weight : node->get_weights()){
			weight = 0.7 * (rand() / (float) RAND_MAX);
			if(j%2)
				weight = - weight;  // make half of the weights negative
			j++;
		}
	}
}


/**
 * Returns the network's classification using the ID of the node with the highest output
 * @return
 */
int neuronal_network::get_network_classification() const{
	layer* layer = get_layer(OUTPUT);
	node* max = layer->get_node(0);
	int maxIndex = 0;

	for(int i = 1; i < layer->get_node_count(); i++){
		node* node = layer->get_node(i);
		if(node->get_output() > max->get_output()){
			max = node;
			maxIndex = i;
		}
	}

	return maxIndex;
}


/**
 * Feeds some data into the input-layer of the network
 * @param input  a vector with the input values
 */
void neuronal_network::feed_input(std::vector<float> input) {
	layer* inputLayer = get_layer(INPUT);

	// copy the input values to the inputLayer
	//parallel
//	int threadCount = std::thread::hardware_concurrency();
//	int elementsPerThread = input.size() / threadCount;
//	std::vector<std::thread> threads(0);
//
//	for(int thID = 0; thID < threadCount; thID++){
//		int rangeFrom = thID * elementsPerThread;
//		int rangeTo = (thID == (threadCount - 1) ? input.size() : ((thID + 1) * elementsPerThread) );
//		threads.push_back(std::thread([rangeFrom, rangeTo, &input, &inputLayer]{
//			// copy the input values to the inputLayer
//			for(int i = rangeFrom; i < rangeTo; i++){
//				Node* inputNode = inputLayer->get_node(i);
//				inputNode->setOutput(input[i]);
//			}
//		}));
//	}
//
//	for(auto& thread : threads)
//		thread.join();

	// sequentiell
	for(int i = 0; i < input.size(); i++){
		node* inputNode = inputLayer->get_node(i);
		inputNode->set_output(input[i]);
	}
}


/**
 * Feeds input layer values forward to the hidden- and then to the output-layer (calculation and activation fct)
 */
void neuronal_network::feed_forward_network() {
	calc_layer(HIDDEN);
	calc_layer(OUTPUT);
}


/**
 * Back propagates network error from output-layer to hidden-layer
 * @param targetClassification  correct classification of the input stream (label)
 */
void neuronal_network::backpropagate_network(int targetClassification) {
	backpropagate_output_layer(targetClassification);
	backpropagate_hidden_layer(targetClassification);
}


/**
 * Updates a node's weights based on given delta
 * @param lType   Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param nodeID  id of the node to update
 * @param delta   difference between desired output and actual output
 */
void neuronal_network::update_node_weights(layer_type lType, int nodeID, float delta){
	layer* updateLayer = get_layer(lType);
	node*  updateNode  = updateLayer->get_node(nodeID);

	layer* prevLayer;
	if(lType == OUTPUT){
		prevLayer = get_layer(HIDDEN);
	}
	else{ // lType == HIDDEN
		prevLayer = get_layer(INPUT);
	}

	// parallel
//	int threadCount = std::thread::hardware_concurrency();
//	int weightsPerThread = updateNode->get_weights().size() / threadCount;
//	std::vector<std::thread> threads(0);
//
//	for(int thID = 0; thID < threadCount; thID++){
//		int rangeFrom = thID * weightsPerThread;
//		int rangeTo = (thID == (threadCount - 1) ? updateNode->get_weights().size()  : ((thID + 1) * weightsPerThread) );
//		threads.push_back(std::thread([this, rangeFrom, rangeTo, delta, &updateNode, &prevLayer]{
//			// go through all weights of updateNode and update them with the delta
//			int i = 0;
//			for(float& weight: updateNode->get_weights()){
//				node* prevLayerNode = prevLayer->get_node(i);
//				weight += (m_learning_rate * prevLayerNode->get_output() * delta);
//				i++;
//			}
//
//			// update bias weigth
//			updateNode->set_bias(updateNode->get_bias() + m_learning_rate * delta);
//		}));
//	}
//
//	for(auto& thread : threads)
//		thread.join();

	// sequentiell
	// go through all weights of updateNode and update them with the delta
	int i = 0;
	for(float& weight: updateNode->get_weights()){
		node* prevLayerNode = prevLayer->get_node(i);
		weight += (m_learning_rate * prevLayerNode->get_output() * delta);
		i++;
	}

	// update bias weigth
	updateNode->set_bias(updateNode->get_bias() + m_learning_rate * delta);
}


/**
 * Performs the SEGMOID activiation function to the specified node
 * @param lType  Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param nodeID id of the node to activate
 */
void neuronal_network::activate_node(layer_type lType, int nodeID) {
	layer* layer = get_layer(lType);
	node*  node  = layer->get_node(nodeID);

	node->set_output(1.0 / (1 + std::exp((float) -1 * node->get_output()))); // SIGMOID activation function
}


/**
 * Calculates the output value of the specified node by multiplying all its weights with the previous layer's outputs
 * @param lType   Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param nodeID  id of the node to calculate
 */
void neuronal_network::calc_node_output(layer_type lType, int nodeID) {
	layer* calcLayer = get_layer(lType);
	node*  calcNode  = calcLayer->get_node(nodeID);

	layer* prevLayer = get_layer(INPUT);
	if(lType == OUTPUT){
		prevLayer = get_layer(HIDDEN);
	}

	float output = calcNode->get_bias();  // start with the nodes bias
	for(int i = 0; i < prevLayer->get_node_count(); i++){
		node* prevLayerNode = prevLayer->get_node(i);
		output += prevLayerNode->get_output() * calcNode->get_weights()[i];
	}
	calcNode->set_output(output);
}


/**
 * Calculates the output values of the given layer
 * @param lType  Type of layer (INPUT, HIDDEN, OUTPUT)
 */
void neuronal_network::calc_layer(layer_type lType) {
	layer* layer = get_layer(lType);

	// parallel
//	int threadCount = std::thread::hardware_concurrency();
//	int nodesPerThread = layer->getNodeCount() / threadCount;
//	std::vector<std::thread> threads(0);
//
//	for(int thID = 0; thID < threadCount; thID++){
//		int rangeFrom = thID * nodesPerThread;
//		int rangeTo = (thID == (threadCount - 1) ? layer->getNodeCount() : ((thID + 1) * nodesPerThread) );
//		threads.push_back(std::thread([this, rangeFrom, rangeTo, lType]{
//			for(int i = rangeFrom; i < rangeTo; i++){
//				calcNodeOutput(lType, i);
//				activateNode(lType, i);
//			}
//		}));
//	}
//
//	for(auto& thread : threads)
//		thread.join();

	// seqentiell
	for(int i = 0; i < layer->get_node_count(); i++){
		calc_node_output(lType, i);
		activate_node(lType, i);
	}
}


/**
 * Back propagates network error to hidden-layer
 * @param targetClassification  correct classification of the input stream (label)
 */
void neuronal_network::backpropagate_hidden_layer(int targetClassification) {
	layer* outputLayer = get_layer(OUTPUT);
	layer* hiddenLayer = get_layer(HIDDEN);

//	int threadCount = std::thread::hardware_concurrency();
//	int nodesPerThread = hiddenLayer->getNodeCount() / threadCount;
//	std::vector<std::thread> threads(0);
//
//	for(int thID = 0; thID < threadCount; thID++){
//		int rangeFrom = thID * nodesPerThread;
//		int rangeTo = (thID == (threadCount - 1) ? hiddenLayer->getNodeCount() : ((thID + 1) * nodesPerThread) );
//		threads.push_back(std::thread([this, rangeFrom, rangeTo, &hiddenLayer, &outputLayer, targetClassification]{
//			for(int i = rangeFrom; i < rangeTo; i++){
//				Node* hiddenNode = hiddenLayer->get_node(i);
//				float outputErrorSum = 0;
//
//				for(int k = 0; k < outputLayer->getNodeCount(); k++){
//					Node* outputNode = outputLayer->get_node(k);
//					int targetOutput = (k == targetClassification) ? 1 : 0;
//
//					float errorDelta  = targetOutput - outputNode->getOutput();
//					float errorSignal = errorDelta *  outputNode->getOutput() * (1 - outputNode->getOutput());	// derivative of the SIGMOID activation function
//					outputErrorSum += errorSignal * outputNode->get_weights()[i];
//				}
//
//				double hiddenErrorSignal = outputErrorSum * hiddenNode->getOutput() * (1 - hiddenNode->getOutput()); // derivative of the SIGMOID activation function
//				updateNodeWeights(HIDDEN, i, hiddenErrorSignal);
//			}
//		}));
//	}
//
//	for(auto& thread : threads)
//		thread.join();

	// sequentiell
	for(int i = 0; i < hiddenLayer->get_node_count(); i++){
		node* hiddenNode = hiddenLayer->get_node(i);
		float outputErrorSum = 0;

		for(int k = 0; k < outputLayer->get_node_count(); k++){
			node* outputNode = outputLayer->get_node(k);
			int targetOutput = (k == targetClassification) ? 1 : 0;

			float errorDelta  = targetOutput - outputNode->get_output();
			float errorSignal = errorDelta *  outputNode->get_output() * (1 - outputNode->get_output());	// derivative of the SIGMOID activation function
			outputErrorSum += errorSignal * outputNode->get_weights()[i];
		}

		double hiddenErrorSignal = outputErrorSum * hiddenNode->get_output() * (1 - hiddenNode->get_output()); // derivative of the SIGMOID activation function
		update_node_weights(HIDDEN, i, hiddenErrorSignal);
	}
}

/**
 * processes the given input with the label and returns the computed classification
 * @param input  the input values
 * @param label  the label of the input
 * @param updateWeights   flag, if the weights should been updated (training)
 * @param usedThreadCount number of threads that should be used
 * @return the computed classification
 */
int neuronal_network::proccess_input(std::vector<float> input, int label, bool updateWeights, int usedThreadCount) {
	std::vector<std::thread> threads(usedThreadCount);

//	for(int thID = 0; thID < usedThreadCount; thID++){
////		int rangeFrom = thID * nodesPerThread;
////		int rangeTo = (thID == (threadCount - 1) ? outputLayer->get_nodeCount() : ((thID + 1) * nodesPerThread) );
//
//
//		threads.push_back(std::thread([this]{
//
//		}));
//	}
//
//	for(auto& thread : threads)
//		thread.join();

	feed_input(input);

	feed_forward_network();

	if(updateWeights)
		backpropagate_network(label);

	return get_network_classification();
}

/**
 * Back propagates network error to output-layer
 * @param targetClassification  correct classification of the input stream (label)
 */
void neuronal_network::backpropagate_output_layer(int targetClassification) {
	layer* outputLayer = get_layer(OUTPUT);

//	int threadCount = std::thread::hardware_concurrency();
//	int nodesPerThread = outputLayer->getNodeCount() / threadCount;
//	std::vector<std::thread> threads(0);
//
//	for(int thID = 0; thID < threadCount; thID++){
//		int rangeFrom = thID * nodesPerThread;
//		int rangeTo = (thID == (threadCount - 1) ? outputLayer->getNodeCount() : ((thID + 1) * nodesPerThread) );
//		threads.push_back(std::thread([this, rangeFrom, rangeTo, &outputLayer, targetClassification]{
//			for(int i = rangeFrom; i < rangeTo; i++){
//				Node* outputNode = outputLayer->get_node(i);
//				int targetOutput = (i == targetClassification) ? 1 : 0;
//
//				float errorDelta  = targetOutput - outputNode->getOutput();
//				float errorSignal = errorDelta *  outputNode->getOutput() * (1 - outputNode->getOutput());	// derivative of the SIGMOID activation function
//
//				updateNodeWeights(OUTPUT, i, errorSignal);
//			}
//		}));
//	}
//
//	for(auto& thread : threads)
//		thread.join();

	// sequentiell
	for(int i = 0; i < outputLayer->get_node_count(); i++){
		node* outputNode = outputLayer->get_node(i);
		int targetOutput = (i == targetClassification) ? 1 : 0;

		float errorDelta  = targetOutput - outputNode->get_output();
		float errorSignal = errorDelta *  outputNode->get_output() * (1 - outputNode->get_output());	// derivative of the SIGMOID activation function

		update_node_weights(OUTPUT, i, errorSignal);
	}
}
