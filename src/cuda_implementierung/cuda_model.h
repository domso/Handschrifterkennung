#ifndef cuda_model_h
#define cuda_model_h

#include <vector>

namespace cuda {

	class model {
	public:
		model();
		model(const model& m) = delete;

		void init(const int size);
		std::vector<float>& getWeights();
		const std::vector<float>& getWeights() const;

	private:
		std::vector<float> weights_;
	};

}

#endif
