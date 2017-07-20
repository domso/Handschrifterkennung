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

namespace cpu {

	/**
	 * types of layers
	 */
	typedef enum layers {INPUT, HIDDEN, OUTPUT} layer_type;

	class layer {
	public:

		/**
		 * constructor
		 */
		layer();

		/**
		 * destructor
		 */
		virtual ~layer();

		/**
		 * @param nodeID  the index of the node
		 * @return the node with the given index
		 */
		node& get_node(const int nodeID);

		/**
		 * @param nodeID  the index of the node
		 * @return the node with the given index
		 */
		const node& get_node(const int nodeID) const;

		/**
		 * @return the number of nodes in this layer
		 */
		int   get_node_count() const;

		/**
		 * inserts the given node in this layer
		 * @param newNode  the node to be inserted
		 */
		void  insert_node(const node& newNode);

	private:
		/**
		 * the nodes of this layer
		 */
		std::vector<node> m_nodes;
	};

}
#endif /* LAYER_H_ */
