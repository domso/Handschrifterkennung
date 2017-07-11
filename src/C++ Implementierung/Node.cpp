/*
 * Node.cpp
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#include "Node.h"

Node::Node(float bias, float output, int weightCount) : weights(weightCount){
	this->bias = bias;
	this->output = output;
}

Node::~Node() {
}

float Node::getOutput() const{
	return output;
}

void Node::setOutput(float output) {
	this->output = output;
}

float Node::getBias() const{
	return bias;
}

void Node::setBias(float bias) {
	this->bias = bias;
}

std::vector<float>& Node::getWeights(){
	return weights;
}
