/*
 * Layer.h
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#ifndef LAYER_H_
#define LAYER_H_
#include <vector>
#include "Node.h"

typedef enum LayerType {INPUT, HIDDEN, OUTPUT} LayerType;

class Layer {
public:
	Layer(int nodeCount);
	virtual ~Layer();
	int getNodeCount();
	Node* getNode(int nodeID) const;
	void  setNode(Node* node, int nodeID);

private:
	std::vector<Node*> nodes;
};

#endif /* LAYER_H_ */
