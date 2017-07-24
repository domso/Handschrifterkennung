#ifndef cuda_kernel_h
#define cuda_kernel_h

/**
 * Kernel for the standard feed-forward to the next layer.
 * Every block represents one node in the next-layer.
 * Every thread represents one edge to the "block"-node.
 *
 * Every thread calculates the weighted edge-value, which
 * will be reduced (sum!) to the root-thread (id = 0)
 *
 * The root calculates the result of the activation function
 * and stores the result in the next-layer
 *
 * All boundaries needs be adjusted on the caller side!
 *
 * The weights are organized in the following form:
 * - every node has (n+1) incoming edges (n = size of input, 1 for the bias)
 * - the weights for node-x start at position x*(n+1)
 *
 * @param input: last calculated layer (only in the first iteration the input-layer!)
 * @param next: next layer
 * @param weights: weights for all edges to the next layer
 */
__global__ void cuda_neural_network(float* input, float* next, float* weights);

/**
 * Kernel for the back-propagation
 * Every block represents one node in the current-layer.
 * Every thread represents one edge from the "block"-node to the next-layer
 *
 * If labels is not null, the next should layer not contain the local error,
 * but the local output. The error of node i will be calculated using the expected
 * output stored in labels[i];
 *
 * Every thread calculates the weighted-error, which will be reduced to the
 * root thread (id = 0)
 *
 * The root calculates the node-error and stores in current, if override is set to true
 * Then he scatters the weighted-error-sum together with the learning rate and the original output
 * to all threads.
 *
 * In the end every thread updates his own edge.
 *
 * All boundaries needs be adjusted on the caller side!
 *
 * @param current: current layer
 * @param next: next layer after current
 * @param weights: weights for all edges from current to next (warning! not the same order! see cuda_neural_network)
 * @param learning: learning_rate
 * @param labels: expected result on the next layer (set to nullptr for next != output)
 * @param override: override the values in current with the node-error
 */
__global__ void cuda_neural_network_error(float* current, float* next,
		float* weights, float* learning, float* labels, bool override);

#endif
