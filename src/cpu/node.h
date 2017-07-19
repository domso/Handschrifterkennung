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
	node(float bias, float output, int weightCount);
	virtual ~node();
	float get_bias() const;
	float get_output() const;
	std::vector<float>& get_weights();
	void set_bias(float bias);
	void set_output(float output);

private:
	float m_bias;
	float m_output;
	std::vector<float> m_weights;
};

}

#endif /* NODE_H_ */
