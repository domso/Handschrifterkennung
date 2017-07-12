#ifndef cuda_model_h
#define cuda_model_h

#include <vector>

namespace cuda {

	class model {
	public:
		model();
		model(const model& m) = delete;

		void init(const int size);
		std::vector<float>& get_weights();
		const std::vector<float>& get_weights() const;

	private:
		std::vector<float> weights_;
	};

}

#endif
