#ifndef cuda_neuronal_network_h
#define cuda_neuronal_network_h

#include <vector>
#include "cuda_model.h"
#include "cuda_ressource.h"
#include "../sample.h"

namespace cuda {
/*
 * Neuronal Network for Handwriting Recognition
 */
class neuronal_network {
public:
	neuronal_network();
	~neuronal_network();
	neuronal_network(const model& m) = delete;

	/*
	 * Parameters for the NN
	 */
	struct config_t {
		float learningRate = 0.15;
		int numHidden = 20;
		int numOutput = 10;
	};

	/*
	 * Updates the configuration
	 * @param config: new configuration
	 */
	void set_config(const config_t config);

	/*
	 * Evaluation/Test Results
	 */
	struct test_result_t {
		int correct = 0;
		int error = 0;
		int total = 0;
		float ratio = 0;
	};

	/*
	 * Trains the given model with the given trainingsData
	 * @param model: untrained-model
	 * @param trainingsData: container with all labeled training-samples
	 * @return: true on success
	 */
	bool train(cuda::model& model, std::vector<data::sample<float>>& trainingsData);

	/*
	 * Tests the given trained model on the given testData
	 * @param model: trained-model
	 * @param testData: container with all labeled testData-samples
	 * @return: basic metrics wrapped in a result-object (filled with -1 on error)
	 */
	test_result_t test(cuda::model& model, std::vector<data::sample<float>>& testData);

	/*
	 * Loads the given model to the graphic-card and set up all required data-structs
	 * @param model: trained-model
	 * @param s: a reference object (not important, just for the size)
	 * @return: true on success
	 */
	bool set_classify_context(cuda::model& model, data::sample<float>& s);

	/*
	 * Classifies the sample with the current context (see set_classify_context)
	 * @param s: unlabeled sample
	 * @return: classification (>=0) or -1 on any error
	 */
	int classify(data::sample<float>& s);
private:

	/*
	 * Internal trainings-context
	 */
	struct train_data_context {
		std::vector<float> hiddenLayer;
		std::vector<float> outputLayer;

		std::vector<float> labels;

		ressource<float> devInput;
		ressource<float> devHidden;
		ressource<float> devOutput;
		ressource<float> devWeights;
		ressource<float> devLabels;
		ressource<float> devLearning;

		/*
		 * Allocates memory for a new context with the given arguments on the graphic-card
		 * @param config: NN-configuration
		 * @param model: untrained-model
		 * @param samples: labeled training-samples
		 */
		train_data_context(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);
		/*
		 * Copies the memory for a new context with the given arguments to the graphic-card
		 * @param config: NN-configuration
		 * @param model: untrained-model
		 * @param samples: labeled training-samples
		 */
		void synchronize(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);

		/*
		 * @return: true, if all operations were successfull
		 */
		bool error_check();
	};

	/*
	 * Internal test-context (subset of train_data_context)
	 */
	struct test_data_context {
		std::vector<float> hiddenLayer;
		std::vector<float> outputLayer;

		ressource<float> devInput;
		ressource<float> devHidden;
		ressource<float> devOutput;
		ressource<float> devWeights;

		/*
		 * Allocates memory for a new context with the given arguments on the graphic-card
		 * @param config: NN-configuration
		 * @param model: untrained-model
		 * @param samples: labeled test-samples
		 */
		test_data_context(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);

		/*
		 * Copies the memory for a new context with the given arguments to the graphic-card
		 * @param config: NN-configuration
		 * @param model: untrained-model
		 * @param samples: labeled test-samples
		 */
		void synchronize(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples);

		/*
		 * @return: true, if all operations were successful
		 */
		bool error_check();
	};

	/*
	 * Performs a single training for a sample
	 * @param i: index inside the trainings-data
	 * @param sample: reference sample for the size
	 * @param context: the current trainings-context
	 * @return: true on success
	 */
	bool train_sample(const int i, const data::sample<float>& sample, train_data_context& context);

	/*
	 * Performs a single test for a sample
	 * @param i: index inside the test-data
	 * @param sample: reference sample for the size
	 * @param context: the current test-context
	 * @return: classification-result on success, -1 on error
	 */
	int test_sample(const int i, const data::sample<float>& sample, test_data_context& context);

	config_t m_currentConfig;
	test_data_context* m_currentContext;
};
}

#endif
