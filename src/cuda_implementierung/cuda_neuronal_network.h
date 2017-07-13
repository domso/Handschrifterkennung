#ifndef cuda_neuronal_network_h
#define cuda_neuronal_network_h

#include <vector>
#include "cuda_model.h"
#include "cuda_helper.h"
#include "../sample.h"

namespace cuda {

	class neuronal_network {
	public:
		neuronal_network();
		~neuronal_network();
		neuronal_network(const model& m) = delete;

		struct config_t {
			float learningRate = 0.15;
			int numHidden = 20;
			int numOutput = 10;
		};

		void set_config(const config_t config);

		struct test_result_t {
			int correct = 0;
			int error = 0;
			int total = 0;
			float ratio = 0;
		};

		void train(cuda::model& model, std::vector<data::sample<float>>& trainingsData);
		test_result_t test(cuda::model& model, std::vector<data::sample<float>>& testData);

		void set_classify_context(cuda::model& model, data::sample<float>& s);
		uint8_t classify(data::sample<float>& s);
	private:

		struct train_data_context {
			std::vector<float> hiddenLayer;
			std::vector<float> outputLayer;

			std::vector<float> labels;

			ressource<float> devInput;
			ressource<float> devHidden;
			ressource<float> devOutput;
			ressource<float> devWeights;
			ressource<float> devLabels;
			ressource<int> devMode;
			ressource<float> devLearning;

			train_data_context(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);
			void synchronize(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);
		};

		struct test_data_context {
			std::vector<float> hiddenLayer;
			std::vector<float> outputLayer;

			ressource<float> devInput;
			ressource<float> devHidden;
			ressource<float> devOutput;
			ressource<float> devWeights;

			test_data_context(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);
			void synchronize(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);
		};

		bool train_sample(const int i, const data::sample<float>& sample, train_data_context& context);
		int test_sample(const int i, const data::sample<float>& sample, test_data_context& context);

		config_t m_currentConfig;
		test_data_context* m_currentContext;
	};
}

#endif
