/**
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

	neuronal_network::neuronal_network(const int inputCount, const int hiddenCount, const int outputCount) {
		srand(time(NULL));
		m_learning_rate = 0.2;

		m_layers.push_back(create_layer(inputCount, 0));
		m_layers.push_back(create_layer(hiddenCount, inputCount));
		m_layers.push_back(create_layer(outputCount, hiddenCount));

		init_weights(HIDDEN);
		init_weights(OUTPUT);
	}

	neuronal_network::~neuronal_network() {}

	int neuronal_network::classify(const data::sample<float>& s) {
		std::vector<data::sample<float>> inputSamples;
		inputSamples.push_back(s);
		proccess_input(inputSamples, false, 8);
		return get_network_classification();
	}

	layer neuronal_network::create_layer(const int nodeCount, const int weightCount) {
		layer l;

		for (int i = 0; i < nodeCount; i++) {
			node n(0.0, 0.0, weightCount);
			l.insert_node(std::move(n));
		}

		return std::move(l);
	}

	layer& neuronal_network::get_layer(const layer_type lType) {
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

	const layer& neuronal_network::get_layer(const layer_type lType) const {
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

	int neuronal_network::proccess_input(const std::vector<data::sample<float>>& inputSamples, const bool updateWeights, const int usedThreadCount) {
		int numError = 0;
		util::barrier barrier(usedThreadCount);
		std::vector<std::thread> threads(0);
		for (int thID = 0; thID < usedThreadCount; thID++) {
			threads.push_back(std::thread([this, &barrier, &inputSamples, thID, usedThreadCount, updateWeights, &numError] {

				for (auto& input :inputSamples) {
					auto label = input.get_label();
					layer& inputLayer = get_layer(INPUT);
					layer& hiddenLayer = get_layer(HIDDEN);
					layer& outputLayer = get_layer(OUTPUT);

					// feed the input into the network
					feed_input(input.internal_data(), usedThreadCount, thID);
					barrier.wait();

					// feed forward through the network
					feed_forward(hiddenLayer, inputLayer, usedThreadCount, thID);
					barrier.wait();
					feed_forward(outputLayer, hiddenLayer, usedThreadCount, thID);
					barrier.wait();

					if(updateWeights) {
						// backpropagate the output and the hidden layer
						backpropagate_output_layer(label, usedThreadCount, thID);
						barrier.wait();
						backpropagate_hidden_layer(label, usedThreadCount, thID);
						barrier.wait();
					}
					if(thID == 0)// first thread checks, if the classification was correct
						numError += get_network_classification() != label;
				}
			}));
		}

		for (auto& thread : threads)
			thread.join();
		return numError;
	}

	void neuronal_network::set_learning_rate(const float learningRate) {
		m_learning_rate = learningRate;
	}

	void neuronal_network::activate_node(node& node) {
		node.set_output(1.0 / (1 + std::exp((float) -1 * node.get_output()))); // SIGMOID activation function
	}

	void neuronal_network::backpropagate_hidden_layer(const int targetClassification, const int usedThreadCount, const int thID) {
		layer& inputLayer  = get_layer(INPUT);
		layer& hiddenLayer = get_layer(HIDDEN);
		layer& outputLayer = get_layer(OUTPUT);

		int elementsPerThread = hiddenLayer.get_node_count() / usedThreadCount;
		int rangeFrom = thID * elementsPerThread;
		int rangeTo = (thID == (usedThreadCount - 1) ? hiddenLayer.get_node_count() : ((thID + 1) * elementsPerThread) );
		for(int i = rangeFrom; i < rangeTo; i++) {
			node& hiddenNode = hiddenLayer.get_node(i);
			float outputErrorSum = 0;

			for(int k = 0; k < outputLayer.get_node_count(); k++) {
				node& outputNode = outputLayer.get_node(k);
				int targetOutput = (k == targetClassification) ? 1 : 0;

				float errorDelta = targetOutput - outputNode.get_output();
				float errorSignal = errorDelta * outputNode.get_output() * (1 - outputNode.get_output());	// derivative of the SIGMOID activation function
				outputErrorSum += errorSignal * outputNode.get_weights()[i];
			}

			double hiddenErrorSignal = outputErrorSum * hiddenNode.get_output() * (1 - hiddenNode.get_output()); // derivative of the SIGMOID activation function
			update_node_weights(hiddenLayer, inputLayer, hiddenNode, hiddenErrorSignal);
		}
	}

	void neuronal_network::backpropagate_output_layer(const int targetClassification, const int usedThreadCount, const int thID) {
		layer& hiddenLayer = get_layer(HIDDEN);
		layer& outputLayer = get_layer(OUTPUT);
		int elementsPerThread = outputLayer.get_node_count() / usedThreadCount;
		int rangeFrom = thID * elementsPerThread;
		int rangeTo = (thID == (usedThreadCount - 1) ? outputLayer.get_node_count() : ((thID + 1) * elementsPerThread) );
		for(int i = rangeFrom; i < rangeTo; i++) {
			node& outputNode = outputLayer.get_node(i);
			int targetOutput = (i == targetClassification) ? 1 : 0;

			float errorDelta = targetOutput - outputNode.get_output();
			float errorSignal = errorDelta * outputNode.get_output() * (1 - outputNode.get_output());	// derivative of the SIGMOID activation function

			update_node_weights(outputLayer, hiddenLayer, outputNode, errorSignal);
		}
	}

	void neuronal_network::calc_node_output(layer& calcLayer, layer& prevLayer, node& calcNode) {
		float output = calcNode.get_bias();  // start with the nodes bias
		for (int i = 0; i < prevLayer.get_node_count(); i++) {
			node& prevLayerNode = prevLayer.get_node(i);
			output += prevLayerNode.get_output() * calcNode.get_weights()[i];
		}
		calcNode.set_output(output);
	}

	void neuronal_network::init_weights(const layer_type lType) {
		layer& layer = get_layer(lType);
		int nodeCount = layer.get_node_count();

		for (int i = 0; i < nodeCount; i++) {
			node& node = layer.get_node(i);
			for (float& weight : node.get_weights()) {
				// init weight with random value in [-0.5, 0.5]
				weight = -0.5 + (float) (std::rand() % 1000001) / (float) 1000000;
			}
		}
	}

	void neuronal_network::feed_forward(layer& actualLayer, layer& prevLayer, const int usedThreadCount, const int thID) {
		int elementsPerThread = actualLayer.get_node_count() / usedThreadCount;
		int rangeFrom = thID * elementsPerThread;
		int rangeTo = (thID == (usedThreadCount - 1) ? actualLayer.get_node_count() : ((thID + 1) * elementsPerThread) );
		for(int i = rangeFrom; i < rangeTo; i++) {
			node& calcNode = actualLayer.get_node(i);
			calc_node_output(actualLayer, prevLayer, calcNode);
			activate_node(calcNode); // activate the node with the SIGMOID activation function
		}
	}

	void neuronal_network::feed_input(const std::vector<float>& input, const int usedThreadCount, const int thID) {
		layer& inputLayer = get_layer(INPUT);

		int elementsPerThread = input.size() / usedThreadCount;
		int rangeFrom = thID * elementsPerThread;
		int rangeTo = (thID == (usedThreadCount - 1) ? input.size() : ((thID + 1) * elementsPerThread));
		for (int i = rangeFrom; i < rangeTo; i++) {
			node& inputNode = inputLayer.get_node(i);
			inputNode.set_output(input[i]);
		}
	}

	void neuronal_network::update_node_weights(layer& actualLayer, layer& prevLayer, node& updateNode, const float delta) {
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
}
