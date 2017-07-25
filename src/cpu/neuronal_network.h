/**
 * neuronal_network.h
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#ifndef NEURONALNETWORK_H_
#define NEURONALNETWORK_H_
#include <vector>
#include "layer.h"
#include "../data/sample.h"

namespace cpu {

class neuronal_network {
public:

	/**
	 * constructor
	 * @param inputCount  Number of nodes in the INPUT layer
	 * @param hiddenCount Number of nodes in the HIDDEN layer
	 * @param outputCount Number of nodes in the OUTPUT layer
	 */
	neuronal_network(const int inputCount, const int hiddenCount, const int outputCount);

	/**
	 * deleted copy constructor
	 */
	neuronal_network(const neuronal_network& o) = delete;

	/**
	 * destructor
	 */
	virtual ~neuronal_network();

	/**
	 * Classifies the given sample (with use of 8 threads)
	 * @param s: unlabeled sample
	 * @return: classification
	 */
	int classify(const data::sample<float>& s);

	/**
	 * Creates a layer
	 * @param nodeCount   the number of nodes that this layer should contain
	 * @param weightCount the number of weights that every node of this layer should contain
	 * @return the created layer
	 */
	layer create_layer(const int nodeCount, const int weightCount);

	/**
	 * Returns the corresponding layer to the given lType of the network
	 * @param lType  Type of layer to be returned (INPUT, HIDDEN, OUTPUT)
	 * @return the corresponding layer to lType
	 */
	layer& get_layer(const layer_type lType);

	/**
	 * Returns the corresponding layer to the given lType of the network
	 * @param lType  Type of layer to be returned (INPUT, HIDDEN, OUTPUT)
	 * @return the corresponding layer to lType
	 */
	const layer& get_layer(const layer_type lType) const;

	/**
	 * @return the network's classification using the index of the node with the highest output
	 */
	int get_network_classification() const;

	/**
	 * processes the given inputSamples and returns the number of false classifications
	 * @param inputSamples     the input samples
	 * @param updateWeights    flag, if the weights should been updated (training)
	 * @param usedThreadCount  number of threads that should be used
	 * @return the number of false computed classifications
	 */
	int proccess_input(const std::vector<data::sample<float>>& inputSamples, const bool updateWeights, const int usedThreadCount);

	/**
	 * sets the learning rate of this network to the given value
	 * @param learningRate  the new learning rate to be set
	 */
	void set_learning_rate(const float learningRate);

private:

	/**
	 * Performs the SEGMOID activation function on the specified node
	 * @param node  the node to activate
	 */
	void activate_node(node& node);

	/**
	 * Back propagates network error to hidden-layer
	 * @param targetClassification  correct classification of the input stream (label)
	 * @param usedThreadCount  number of threads to use
	 * @param thID  id of the actual thread (0 if there is only 1 thread)
	 */
	void backpropagate_hidden_layer(const int targetClassification, const int usedThreadCount, const int thID);

	/**
	 * Back propagates network error to output-layer
	 * @param targetClassification  correct classification of the input stream (label)
	 * @param usedThreadCount  number of threads to use
	 * @param thID  id of the actual thread (0 if there is only 1 thread)
	 */
	void backpropagate_output_layer(const int targetClassification, const int usedThreadCount, const int thID);

	/**
	 * Calculates the output value of the specified node by multiplying all its weights with the previous layer's outputs
	 * @param calcLayer  the layer of the node
	 * @param prevLayer  the previous layer of the node
	 * @param calcNode   the node to calculate
	 */
	void calc_node_output(layer& calcLayer, layer& prevLayer, node& calcNode);

	/**
	 * Initializes a layer's weights with random values between -0.5 and 0.5
	 * @param lType  Defining what layer to initialize
	 */
	void init_weights(const layer_type lType);

	/**
	 * feeds the data of the previous layer forward to the actual layer
	 * @param actualLayer  the actual layer
	 * @param prevLayer    the previous layer
	 * @param usedThreadCount  the number of threads to use
	 * @param thID  id of the actual thread (0 if there is only 1 thread)
	 */
	void feed_forward(layer& actualLayer, layer& prevLayer, const int usedThreadCount, const int thID);

	/**
	 * Feeds the given data into the input-layer of the network
	 * @param input  a vector with the input values
	 * @param usedThreadCount  the number of threads to use
	 * @param thID  id of the actual thread (0 if there is only 1 thread)
	 */
	void feed_input(const std::vector<float>& input, const int usedThreadCount, const int thID);

	/**
	 * Updates a node's weights based on the given delta
	 * @param lType   Type of layer (INPUT, HIDDEN, OUTPUT)
	 * @param nodeID  id of the node to be updated
	 * @param delta   difference between desired output and actual output
	 */
	void update_node_weights(layer& actualLayer, layer& prevLayer, node& updateNode, const float delta);

	/**
	 * the learning rate of the neuronal network
	 */
	float m_learning_rate;

	/**
	 * the layers of this network
	 */
	std::vector<layer> m_layers;
};

}
#endif /** NEURONALNETWORK_H_ */
