/*
 * neuronal_network.h
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#ifndef NEURONALNETWORK_H_
#define NEURONALNETWORK_H_
#include <vector>
#include "layer.h"
#include "../sample.h"

class neuronal_network {
public:
	neuronal_network(int inputCount, int hiddenCount, int outputCount);
	virtual ~neuronal_network();
	neuronal_network(neuronal_network& o) = delete;
	layer create_layer(int nodeCount, int weightCount);
	layer& get_layer(layer_type lType);
	const layer& get_layer(layer_type lType) const;
	int    get_network_classification() const;
	int    proccess_input(std::vector<data::sample<float>>& inputSamples, bool updateWeights, int usedThreadCount);

private:
	void activate_node(layer_type lType, int nodeID);
	void backpropagate_network(int targetClassification);
	void backpropagate_hidden_layer(int targetClassification);
	void backpropagate_output_layer(int targetClassification);
	void calc_layer(layer_type lType);
	void calc_node_output(layer_type lType, int nodeID);
	void init_weights(layer_type lType);
	void feed_forward_network();
	void feed_forward(layer& actualLayer, layer& prevLayer, const int usedThreadCount, const int thID);
	void feed_input(std::vector<float> input);
	void update_node_weights(layer_type lType, int nodeID, float delta);

	float m_learning_rate;
	std::vector<layer> m_layers;
};

#endif /* NEURONALNETWORK_H_ */
