/*
 * NeuronalNetwork.h
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#ifndef NEURONALNETWORK_H_
#define NEURONALNETWORK_H_
#include <vector>
#include "Layer.h"

class NeuronalNetwork {
public:
	NeuronalNetwork(int inputCount, int hiddenCount, int outputCount);
	virtual ~NeuronalNetwork();
	Layer* createInputLayer(int inputCount);
	Layer* createLayer(int nodeCount, int weightCount);
	Layer* getLayer(LayerType lType);
	void initWeights(LayerType lType);
	int  getNetworkClassification();
	void feedInput(std::vector<float> input);
	void feedForwardNetwork();
	void backPropagateNetwork(int targetClassification);
	void updateNodeWeights(LayerType lType, int nodeID, float delta);
	void activateNode(LayerType lType, int nodeID);
	void calcNodeOutput(LayerType lType, int nodeID);
	void calcLayer(LayerType lType);

private:
	void backPropagateHiddenLayer(int targetClassification);
	void backPropagateOutputLayer(int targetClassification);

	float learningRate;
	std::vector<Layer*> layers;
};

#endif /* NEURONALNETWORK_H_ */
