#include <algorithm>
#include "cuda_neuronal_network.h"
#include "cuda_kernel.h"

namespace cuda {

neuronal_network::neuronal_network() :
		m_currentContext(nullptr) {

}

neuronal_network::~neuronal_network() {
	if (m_currentContext != nullptr) {
		delete m_currentContext;
	}
}

void neuronal_network::set_config(const config_t config) {
	m_currentConfig = config;
}

void neuronal_network::train(cuda::model& model,
		std::vector<data::sample<float>>& trainingsData) {
	std::vector<float>& refInput = trainingsData[0].internal_data();

	model.init(
			(refInput.size() + 1) * m_currentConfig.numHidden
					+ (m_currentConfig.numHidden + 1)
							* m_currentConfig.numOutput);

	train_data_context context(m_currentConfig, model, trainingsData);
	context.synchronize(m_currentConfig, model, trainingsData);
	cudaThreadSynchronize();

	for (int i = 0; i < trainingsData.size(); i++) {
		train_sample(i, trainingsData[i], context);
	}

	context.devWeights.synch_from_device(model.get_weights());
}

neuronal_network::test_result_t neuronal_network::test(cuda::model& model,
		std::vector<data::sample<float>>& testData) {
	test_data_context context(m_currentConfig, model, testData);
	context.synchronize(m_currentConfig, model, testData);
	cudaThreadSynchronize();

	test_result_t result;
	int i = 0;

	for (data::sample<float>& s : testData) {
		if (test_sample(i, s, context) == s.get_label()) {
			result.correct++;
		}

		i++;
	}

	result.total = testData.size();
	result.error = result.total - result.correct;
	result.ratio = (float) result.correct / (float) result.total;

	return result;
}

void neuronal_network::set_classify_context(cuda::model& model,
		data::sample<float>& s) {

	if (m_currentContext != nullptr) {
		delete m_currentContext;
	}

	std::vector<data::sample<float>> dummy_data;
	dummy_data.push_back(s);
	m_currentContext = new test_data_context(m_currentConfig, model,
			dummy_data);
	m_currentContext->synchronize(m_currentConfig, model, dummy_data);
}

uint8_t neuronal_network::classify(data::sample<float>& s) {
	if (m_currentContext != nullptr) {
		m_currentContext->devInput.synch_to_device(s.internal_data(), 0);
		cudaThreadSynchronize();
		return test_sample(0, s, *m_currentContext);
	}

	return 0;
}

bool neuronal_network::train_sample(const int i,
		const data::sample<float>& sample, train_data_context& context) {
	int num_blocks;
	int num_threads;

	num_blocks = context.hiddenLayer.size();
	num_threads = sample.size();
	cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>((&context.devInput) + i * sample.size(), &context.devHidden, &context.devWeights);
	//cudaThreadSynchronize();

	if (cudaSuccess != cudaGetLastError()) {
		return false;
	}

	num_blocks = context.outputLayer.size();
	num_threads = context.hiddenLayer.size();
	cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&context.devHidden, &context.devOutput, (&context.devWeights) + ((sample.size() + 1) * context.hiddenLayer.size()));
	//cudaThreadSynchronize();

	if (cudaSuccess != cudaGetLastError()) {
		return false;
	}

	num_blocks = 1;
	num_threads = context.outputLayer.size();
	cuda_neural_network_output_error<<<num_blocks, num_threads>>>(&context.devOutput, (&context.devLabels) + i * context.outputLayer.size());
	//cudaThreadSynchronize();

	if (cudaSuccess != cudaGetLastError()) {
		return false;
	}

	num_blocks = context.hiddenLayer.size();
	num_threads = context.outputLayer.size();
	cuda_neural_network_error<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&context.devHidden, &context.devOutput, (&context.devWeights) + ((sample.size() + 1) * context.hiddenLayer.size()), &context.devLearning, &context.devMode);
	//cudaThreadSynchronize();

	if (cudaSuccess != cudaGetLastError()) {
		return false;
	}

	num_blocks = sample.size();
	num_threads = context.hiddenLayer.size();
	cuda_neural_network_error<<<num_blocks, num_threads, num_threads * sizeof(float)>>>((&context.devInput) + i * sample.size(), &context.devHidden, &context.devWeights, &context.devLearning, &context.devMode);
	//cudaThreadSynchronize();

	if (cudaSuccess != cudaGetLastError()) {
		return false;
	}

	return true;
}

int neuronal_network::test_sample(const int i,
		const data::sample<float>& sample, test_data_context& context) {
	int num_blocks;
	int num_threads;

	num_blocks = context.hiddenLayer.size();
	num_threads = sample.size();
	cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>((&context.devInput) + i * sample.size(), &context.devHidden, &context.devWeights);
	//cudaThreadSynchronize();

	if (cudaSuccess != cudaGetLastError()) {
		return false;
	}

	num_blocks = context.outputLayer.size();
	num_threads = context.hiddenLayer.size();
	cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&context.devHidden, &context.devOutput, (&context.devWeights) + ((sample.size() + 1) * context.hiddenLayer.size()));
	//cudaThreadSynchronize();

	if (cudaSuccess != cudaGetLastError()) {
		return false;
	}

	context.devOutput.synch_from_device(context.outputLayer);
	cudaThreadSynchronize();

	return std::distance(context.outputLayer.begin(),
			std::max_element(context.outputLayer.begin(),
					context.outputLayer.end()));
}

neuronal_network::train_data_context::train_data_context(const config_t config,
		const cuda::model& model,
		const std::vector<data::sample<float>>& samples) :
		hiddenLayer(config.numHidden), outputLayer(config.numOutput), labels(
				samples.size() * outputLayer.size()), devInput(
				samples[0].internal_data(), samples.size()), devHidden(
				hiddenLayer), devOutput(outputLayer), devWeights(
				model.get_weights()), devLabels(labels) {
	for (int i = 0; i < samples.size(); i++) {
		for (int j = 0; j < outputLayer.size(); j++) {
			if (samples[i].get_label() == j) {
				labels[i * outputLayer.size() + j] = 1;
			} else {
				labels[i * outputLayer.size() + j] = 0;
			}
		}
	}
}

void neuronal_network::train_data_context::synchronize(const config_t config,
		const cuda::model& model,
		const std::vector<data::sample<float>>& samples) {

	std::vector<float> combinedData(
			samples.size() * samples[0].internal_data().size());
	int index = 0;

	for (auto& s : samples) {
		for (auto& f : s.internal_data()) {
			combinedData[index] = f;
			index++;
		}
	}

	devInput.synch_to_device(combinedData);

	devHidden.synch_to_device(hiddenLayer);
	devOutput.synch_to_device(outputLayer);
	devWeights.synch_to_device(model.get_weights());
	devLabels.synch_to_device(labels);
	devLearning.synch_to_device(config.learningRate);
	devMode.synch_to_device(1);

	error_check();
}

bool neuronal_network::train_data_context::error_check() {
	bool error = false;
	error |= devInput.has_error();
	error |= devHidden.has_error();
	error |= devOutput.has_error();
	error |= devWeights.has_error();
	error |= devLabels.has_error();
	error |= devMode.has_error();
	error |= devLearning.has_error();

	return error;
}

neuronal_network::test_data_context::test_data_context(const config_t config,
		const cuda::model& model,
		const std::vector<data::sample<float>>& samples) :
		hiddenLayer(config.numHidden), outputLayer(config.numOutput), devInput(
				samples[0].internal_data(), samples.size()), devHidden(
				hiddenLayer), devOutput(outputLayer), devWeights(
				model.get_weights()) {

}

void neuronal_network::test_data_context::synchronize(const config_t config,
		const cuda::model& model,
		const std::vector<data::sample<float>>& samples) {
	std::vector<float> combinedData(
			samples.size() * samples[0].internal_data().size());
	int index = 0;

	for (auto& s : samples) {
		for (auto& f : s.internal_data()) {
			combinedData[index] = f;
			index++;
		}
	}

	devInput.synch_to_device(combinedData);

	devHidden.synch_to_device(hiddenLayer);
	devOutput.synch_to_device(outputLayer);
	devWeights.synch_to_device(model.get_weights());

	error_check();
}

bool neuronal_network::test_data_context::error_check() {
	bool error = false;
	error |= devInput.has_error();
	error |= devHidden.has_error();
	error |= devOutput.has_error();
	error |= devWeights.has_error();

	return error;
}

}
