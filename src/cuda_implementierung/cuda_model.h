#ifndef cuda_model_h
#define cuda_model_h

#include <vector>

namespace cuda {
/*
 * Storage-Class for the (un)trained model
 */
class model {
public:
	model();
	model(const model& m) = delete;

	/*
	 * Initializes the model with the given argument
	 * @param size: number of weights
	 */
	void init(const int size);

	/*
	 * @return: the internal vector containing the weights
	 */
	std::vector<float>& get_weights();

	/*
	 * @return: the internal vector containing the weights
	 */
	const std::vector<float>& get_weights() const;

private:
	std::vector<float> m_weights;
};
}

#endif
