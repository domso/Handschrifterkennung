/*
 * layer.h
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#ifndef LAYER_H_
#define LAYER_H_
#include <vector>
#include "node.h"

typedef enum layers {INPUT, HIDDEN, OUTPUT} layer_type;

class layer {
public:
	layer();
	virtual ~layer();
	node& get_node(const int nodeID);
	const node& get_node(const int nodeID) const;
	int   get_node_count() const;
	void  insert_node(const node& newNode);

private:
	std::vector<node> m_nodes;
};

#endif /* LAYER_H_ */
