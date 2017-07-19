#include "cuda_model.h"
#include <random>
#include <ctime>

namespace cuda {

	model::model() {

	}

	void model::init(const int size) {
		m_weights.resize(size);
		std::srand(std::time(0));

		for (float& w : m_weights) {
			w = -0.5 + (float) (std::rand() % 1000001) / (float) 1000000;;
		}
	}

	std::vector<float>& model::get_weights() {
		return m_weights;
	}

	const std::vector<float>& model::get_weights() const{
			return m_weights;
	}

}
