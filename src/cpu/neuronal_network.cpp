/*
 * neuronal_network.cpp
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#include "neuronal_network.h"
#include "../util/barrier.hpp"
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <iostream>

namespace cpu {
/**
 * @param inputCount  Number of nodes in the INPUT layer
 * @param hiddenCount Number of nodes in the HIDDEN layer
 * @param outputCount Number of nodes in the OUTPUT layer
 */
neuronal_network::neuronal_network(int inputCount, int hiddenCount, int outputCount) {
	srand(time(NULL));
	m_learning_rate = 0.2;

	m_layers.push_back(create_layer(inputCount, 0));
	m_layers.push_back(create_layer(hiddenCount, inputCount));
	m_layers.push_back(create_layer(outputCount, hiddenCount));

	init_weights(HIDDEN);
	init_weights(OUTPUT);
}

neuronal_network::~neuronal_network() {}

/**
 * Returns the corresponding layer to the given lType of the network
 * @param lType  Type of layer to be returned (INPUT, HIDDEN, OUTPUT)
 * @return the corresponding layer to lType
 */
layer& neuronal_network::get_layer(layer_type lType) {
	switch (lType) {
	case INPUT: {
		return m_layers[0];
	}
	case HIDDEN: {
		return m_layers[1];
	}
	case OUTPUT: {
		return m_layers[2];
	}
	default:
		return m_layers[0];
	}
}

/**
 * Returns the corresponding layer to the given lType of the network
 * @param lType  Type of layer to be returned (INPUT, HIDDEN, OUTPUT)
 * @return the corresponding layer to lType
 */
const layer& neuronal_network::get_layer(layer_type lType) const {
	switch (lType) {
	case INPUT: {
		return m_layers[0];
	}
	case HIDDEN: {
		return m_layers[1];
	}
	case OUTPUT: {
		return m_layers[2];
	}
	default:
		return m_layers[0];
	}
}


/**
 * Creates a layer
 * @param nodeCount   the number of nodes that this layer should contain
 * @param weightCount the number of weights that should every node contain
 * @return the created layer
 */
layer neuronal_network::create_layer(int nodeCount, int weightCount) {
	layer l;

	for (int i = 0; i < nodeCount; i++) {
		node n(0.0, 0.0, weightCount);
		l.insert_node(std::move(n));
	}

	return std::move(l);
}

/**
 * Initializes a layer's weights with random values
 * @param lType  Defining what layer to initialize
 */
void neuronal_network::init_weights(layer_type lType) {
	layer& layer = get_layer(lType);
	int nodeCount = layer.get_node_count();

	for (int i = 0; i < nodeCount; i++) {
		node& node = layer.get_node(i);

		int j = 0;
		for (float& weight : node.get_weights()) {
			weight = 0.7 * (rand() / (float) RAND_MAX);
			if (j % 2)
				weight = -weight;  // make half of the weights negative
			j++;
		}
	}
}

/**
 * Returns the network's classification using the ID of the node with the highest output
 * @return the network's classification
 */
int neuronal_network::get_network_classification() const {
	const layer& layer = get_layer(OUTPUT);
	int maxIndex = 0;

	for (int i = 1; i < layer.get_node_count(); i++) {
		const node& currentNode = layer.get_node(i);
		const node& max = layer.get_node(maxIndex);
		if (currentNode.get_output() > max.get_output()) {
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
	layer& inputLayer = get_layer(INPUT);
	for (int i = 0; i < input.size(); i++) {
		node& inputNode = inputLayer.get_node(i);
		inputNode.set_output(input[i]);
	}
}

/**
 *
 * @param actualLayer
 * @param prevLayer
 * @param usedThreadCount
 * @param thID
 */
void neuronal_network::feed_forward(layer& actualLayer, layer& prevLayer, const int usedThreadCount, const int thID) {
	int elementsPerThread = actualLayer.get_node_count() / usedThreadCount;
	int rangeFrom = thID * elementsPerThread;
	int rangeTo = (thID == (usedThreadCount - 1) ? actualLayer.get_node_count() : ((thID + 1) * elementsPerThread) );
	for(int i = rangeFrom; i < rangeTo; i++) {
		//#############################################
		// calc_node_output(HIDDEN, i);
		//#############################################

		node& calcNode = actualLayer.get_node(i);

		float output = calcNode.get_bias();  // start with the nodes bias
		for(int j = 0; j < prevLayer.get_node_count(); j++) {
			node& prevLayerNode = prevLayer.get_node(j);
			output += prevLayerNode.get_output() * calcNode.get_weights()[j];
		}
		calcNode.set_output(output);

		//#############################################
		// activate_node(HIDDEN, i);
		//#############################################

		calcNode.set_output(1.0 / (1 + std::exp((float) -1 * calcNode.get_output())));// SIGMOID activation function
	}
}

/**
 * Feeds input layer values forward to the hidden- and then to the output-layer (calculation and activation fct)
 */
//void neuronal_network::feed_forward_network() {
//	calc_layer(HIDDEN);
//	calc_layer(OUTPUT);
//}

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
void neuronal_network::update_node_weights(layer_type lType, int nodeID, float delta) {
	layer& updateLayer = get_layer(lType);
	node& updateNode = updateLayer.get_node(nodeID);

	layer_type prevLayerType;
	if (lType == OUTPUT) {
		prevLayerType = HIDDEN;
	} else { // lType == HIDDEN
		prevLayerType = INPUT;
	}

	layer& prevLayer = get_layer(prevLayerType);

	// go through all weights of updateNode and update them with the delta
	int i = 0;
	for (float& weight : updateNode.get_weights()) {
		node& prevLayerNode = prevLayer.get_node(i);
		weight += (m_learning_rate * prevLayerNode.get_output() * delta);
		i++;
	}

	// update bias weigth
	updateNode.set_bias(updateNode.get_bias() + m_learning_rate * delta);
}

/**
 * Performs the SEGMOID activiation function to the specified node
 * @param lType  Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param nodeID id of the node to activate
 */
//void neuronal_network::activate_node(layer_type lType, int nodeID) {
//	layer& layer = get_layer(lType);
//	node& node = layer.get_node(nodeID);
//
//	node.set_output(1.0 / (1 + std::exp((float) -1 * node.get_output()))); // SIGMOID activation function
//}

/**
 * Calculates the output value of the specified node by multiplying all its weights with the previous layer's outputs
 * @param lType   Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param nodeID  id of the node to calculate
 */
void neuronal_network::calc_node_output(layer_type lType, int nodeID) {
	layer& calcLayer = get_layer(lType);
	node& calcNode = calcLayer.get_node(nodeID);

	layer& prevLayer = get_layer(INPUT);
	if (lType == OUTPUT) {
		prevLayer = get_layer(HIDDEN);
	}

	float output = calcNode.get_bias();  // start with the nodes bias
	for (int i = 0; i < prevLayer.get_node_count(); i++) {
		node& prevLayerNode = prevLayer.get_node(i);
		output += prevLayerNode.get_output() * calcNode.get_weights()[i];
	}
	calcNode.set_output(output);
}

/**
 * Calculates the output values of the given layer
 * @param lType  Type of layer (INPUT, HIDDEN, OUTPUT)
 */
//void neuronal_network::calc_layer(layer_type lType) {
//	layer& layer = get_layer(lType);
//	for (int i = 0; i < layer.get_node_count(); i++) {
//		calc_node_output(lType, i);
//		activate_node(lType, i);
//	}
//}

/**
 * Back propagates network error to hidden-layer
 * @param targetClassification  correct classification of the input stream (label)
 */
void neuronal_network::backpropagate_hidden_layer(int targetClassification) {
	layer& outputLayer = get_layer(OUTPUT);
	layer& hiddenLayer = get_layer(HIDDEN);
	for (int i = 0; i < hiddenLayer.get_node_count(); i++) {
		node& hiddenNode = hiddenLayer.get_node(i);
		float outputErrorSum = 0;

		for (int k = 0; k < outputLayer.get_node_count(); k++) {
			node& outputNode = outputLayer.get_node(k);
			int targetOutput = (k == targetClassification) ? 1 : 0;

			float errorDelta = targetOutput - outputNode.get_output();
			float errorSignal = errorDelta * outputNode.get_output() * (1 - outputNode.get_output());	// derivative of the SIGMOID activation function
			outputErrorSum += errorSignal * outputNode.get_weights()[i];
		}

		double hiddenErrorSignal = outputErrorSum * hiddenNode.get_output() * (1 - hiddenNode.get_output()); // derivative of the SIGMOID activation function
		update_node_weights(HIDDEN, i, hiddenErrorSignal);
	}
}

/**
 * processes the given inputSamples and returns the number of false calssifications
 * @param inputSamples     the input samples
 * @param updateWeights    flag, if the weights should been updated (training)
 * @param usedThreadCount  number of threads that should be used
 * @return the number of false computed classifications
 */
int neuronal_network::proccess_input(std::vector<data::sample<float>>& inputSamples, bool updateWeights, int usedThreadCount) {
	int numError = 0;
	Barrier barrier(usedThreadCount);
	std::vector<std::thread> threads(0);
	for (int thID = 0; thID < usedThreadCount; thID++) {
		threads.push_back(std::thread([this, &barrier, &inputSamples, thID, usedThreadCount, updateWeights, &numError] {

			for (auto& input :inputSamples) {
				auto label = input.get_label();
				layer& inputLayer = get_layer(INPUT);
				layer& hiddenLayer = get_layer(HIDDEN);
				layer& outputLayer = get_layer(OUTPUT);
				int rangeFrom = 0;
				int rangeTo = 0;
				int elementsPerThread = 0;

				//#############################################
				// feed_input(input);
				//#############################################

				// copy the input values to the inputLayer
				elementsPerThread = input.size() / usedThreadCount;
				rangeFrom = thID * elementsPerThread;
				rangeTo = (thID == (usedThreadCount - 1) ? input.size() : ((thID + 1) * elementsPerThread));

				// copy the input values to the inputLayer
				for(int i = rangeFrom; i < rangeTo; i++) {
					node& inputNode = inputLayer.get_node(i);
					inputNode.set_output(input[i]);
				}

				barrier.wait();

				//#############################################
				// feed_forward_network();
				//#############################################

				feed_forward(hiddenLayer, inputLayer, usedThreadCount, thID);
				barrier.wait();
				feed_forward(outputLayer, hiddenLayer, usedThreadCount, thID);

				if(updateWeights) {

					barrier.wait();

					//#############################################
					// backpropagate_network(label);
					//#############################################
					elementsPerThread = outputLayer.get_node_count() / usedThreadCount;
					rangeFrom = thID * elementsPerThread;
					rangeTo = (thID == (usedThreadCount - 1) ? outputLayer.get_node_count() : ((thID + 1) * elementsPerThread) );
					for(int i = rangeFrom; i < rangeTo; i++) {
						node& outputNode = outputLayer.get_node(i);
						int targetOutput = (i == label) ? 1 : 0;

						float errorDelta = targetOutput - outputNode.get_output();
						float errorSignal = errorDelta * outputNode.get_output() * (1 - outputNode.get_output());	// derivative of the SIGMOID activation function

						update_node_weights(OUTPUT, i, errorSignal);
					}

					barrier.wait();

					elementsPerThread = hiddenLayer.get_node_count() / usedThreadCount;
					rangeFrom = thID * elementsPerThread;
					rangeTo = (thID == (usedThreadCount - 1) ? hiddenLayer.get_node_count() : ((thID + 1) * elementsPerThread) );
					for(int i = rangeFrom; i < rangeTo; i++) {
						node& hiddenNode = hiddenLayer.get_node(i);
						float outputErrorSum = 0;

						for(int k = 0; k < outputLayer.get_node_count(); k++) {
							node& outputNode = outputLayer.get_node(k);
							int targetOutput = (k == label) ? 1 : 0;

							float errorDelta = targetOutput - outputNode.get_output();
							float errorSignal = errorDelta * outputNode.get_output() * (1 - outputNode.get_output());	// derivative of the SIGMOID activation function
							outputErrorSum += errorSignal * outputNode.get_weights()[i];
						}

						double hiddenErrorSignal = outputErrorSum * hiddenNode.get_output() * (1 - hiddenNode.get_output()); // derivative of the SIGMOID activation function
						update_node_weights(HIDDEN, i, hiddenErrorSignal);
					}
				}

				barrier.wait();
				if(thID == 0)// only once
					numError += get_network_classification() != label;
			}
		}));
	}

	for (auto& thread : threads)
		thread.join();
	return numError;
}


/**
 * Back propagates network error to output-layer
 * @param targetClassification  correct classification of the input stream (label)
 */
void neuronal_network::backpropagate_output_layer(int targetClassification) {
	layer& outputLayer = get_layer(OUTPUT);
	for (int i = 0; i < outputLayer.get_node_count(); i++) {
		node& outputNode = outputLayer.get_node(i);
		int targetOutput = (i == targetClassification) ? 1 : 0;

		float errorDelta = targetOutput - outputNode.get_output();
		float errorSignal = errorDelta * outputNode.get_output() * (1 - outputNode.get_output());	// derivative of the SIGMOID activation function

		update_node_weights(OUTPUT, i, errorSignal);
	}
}

}
