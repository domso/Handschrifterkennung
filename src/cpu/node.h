/*
 * node.h
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#ifndef NODE_H_
#define NODE_H_
#include <vector>

namespace cpu {
	class node {
	public:
		/**
		 * Constructor
		 * @param bias  the bias of this node
		 * @param output  the output of this node
		 * @param weightCount  the number of weights of this node
		 */
		node(float bias, float output, int weightCount);

		/**
		 * destructor
		 */
		virtual ~node();

		/**
		 * @return the bias of this node
		 */
		float get_bias() const;

		/**
		 * @return the output of this node
		 */
		float get_output() const;

		/**
		 * @return the reference of the vector with all weights of this node
		 */
		std::vector<float>& get_weights();

		/**
		 * sets the bias of this node to the given value
		 * @param bias  the value to be set
		 */
		void set_bias(float bias);

		/**
		 *sets the output of this node to the given value
		 * @param output  the value to be set
		 */
		void set_output(float output);

	private:
		/**
		 * the bias of this node
		 */
		float m_bias;

		/**
		 * the output of this node
		 */
		float m_output;

		/**
		 * the weights of this node
		 */
		std::vector<float> m_weights;
	};
}

#endif /* NODE_H_ */
