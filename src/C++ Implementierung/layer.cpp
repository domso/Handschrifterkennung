/*
 * layer.cpp
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#include "layer.h"

layer::layer(int nodeCount) : m_nodes(nodeCount){
}

layer::~layer() {
	for(node* n: m_nodes){
		delete n;
	}
}

int layer::get_node_count() const{
	return m_nodes.size();
}

node* layer::get_node(int nodeID) const {
	return m_nodes[nodeID];
}

void layer::set_node(node* node, int nodeID) {
	m_nodes[nodeID] = node;
}
