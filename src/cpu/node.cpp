/*
 * Node.cpp
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#include "node.h"

namespace cpu {

node::node(float bias, float output, int weightCount) : m_weights(weightCount){
	this->m_bias = bias;
	this->m_output = output;
}

node::~node() {
}

float node::get_output() const{
	return m_output;
}

void node::set_output(float output) {
	this->m_output = output;
}

float node::get_bias() const{
	return m_bias;
}

void node::set_bias(float bias) {
	this->m_bias = bias;
}

std::vector<float>& node::get_weights(){
	return m_weights;
}

}
