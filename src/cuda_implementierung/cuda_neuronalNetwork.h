#ifndef cuda_neuronalNetwork_h
#define cuda_neuronalNetwork_h

#include <vector>
#include "cuda_model.h"
#include "cuda_helper.h"
#include "../sample.h"

namespace cuda {

	class neuronalNetwork {
	public:
		neuronalNetwork();
		~neuronalNetwork();
		neuronalNetwork(const model& m) = delete;

		struct config_t {
			float learning_rate = 0.15;
			int num_hidden = 20;
			int num_output = 10;
		};

		void set_config(const config_t config);

		struct test_result_t {
			int correct = 0;
			int error = 0;
			int total = 0;
			float ratio = 0;
		};

		void train(cuda::model& model, std::vector<data::sample<float>>& trainings_data);
		test_result_t test(cuda::model& model, std::vector<data::sample<float>>& test_data);

		void set_classify_context(cuda::model& model, data::sample<float>& s);
		uint8_t classify(data::sample<float>& s);
	private:

		class train_data_context {
		public:
			std::vector<float> hidden_layer;
			std::vector<float> output_layer;

			std::vector<float> labels;

			ressource<float> dev_input;
			ressource<float> dev_hidden;
			ressource<float> dev_output;
			ressource<float> dev_weights;
			ressource<float> dev_labels;
			ressource<int> dev_mode;
			ressource<float> dev_learning;

			train_data_context(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);
			void synchronize(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);
		};

		class test_data_context {
		public:
			std::vector<float> hidden_layer;
			std::vector<float> output_layer;

			ressource<float> dev_input;
			ressource<float> dev_hidden;
			ressource<float> dev_output;
			ressource<float> dev_weights;

			test_data_context(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);
			void synchronize(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);
		};

		bool train_sample(const int i, const data::sample<float>& sample, train_data_context& context);
		int test_sample(const int i, const data::sample<float>& sample, test_data_context& context);

		config_t current_config_;
		test_data_context* current_context_;
	};

}

#endif
