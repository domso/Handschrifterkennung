/*
 * Node.h
 *
 *  Created on: 30.06.2017
 *      Author: haugchri
 */

#ifndef NODE_H_
#define NODE_H_
#include <vector>

class Node {
public:
	Node(float bias, float output, int weightCount);
	virtual ~Node();
	float getOutput() const;
	void setOutput(float output);
	float getBias() const;
	void setBias(float bias);
	std::vector<float>& getWeights();

private:
	float bias;
	float output;
	std::vector<float> weights;
};

#endif /* NODE_H_ */
