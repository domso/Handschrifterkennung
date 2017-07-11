/*
 * Layer.cpp
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#include "Layer.h"

Layer::Layer(int nodeCount) : nodes(nodeCount){
}

Layer::~Layer() {
	for(Node* n: nodes){
		delete n;
	}
}

int Layer::getNodeCount() {
	return nodes.size();
}

Node* Layer::getNode(int nodeID) const {
	return nodes[nodeID];
}

void Layer::setNode(Node* node, int nodeID) {
	nodes[nodeID] = node;
}
