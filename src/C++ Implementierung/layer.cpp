/*
 * layer.cpp
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#include "layer.h"

layer::layer(const int nodeCount) {
}

layer::~layer() {

}

int layer::get_node_count() const{
	return m_nodes.size();
}

node& layer::get_node(const int nodeID) {
	return m_nodes[nodeID];
}

const node& layer::get_node(const int nodeID) const{
	return m_nodes[nodeID];
}

void layer::insert_node(const node& newNode) {
	m_nodes.push_back(newNode);
}
