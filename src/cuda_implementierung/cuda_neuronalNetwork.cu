#include <algorithm>
#include "cuda_neuronalNetwork.h"
#include "cuda_kernel.h"

namespace cuda {

	neuronalNetwork::neuronalNetwork() {

	}

	void neuronalNetwork::set_config(const config_t config) {
		current_config_ = config;
	}

	void neuronalNetwork::train(cuda::model& model, std::vector<data::sample<float>>& trainings_data) {
		std::vector<float>& ref_input = trainings_data[0].internalData();

		model.init((ref_input.size() + 1) * current_config_.num_hidden
				+ (current_config_.num_hidden + 1) * current_config_.num_output);

		data_context context(current_config_, model, trainings_data);
		context.synchronize(current_config_, model, trainings_data);

		int i = 0;

		for (data::sample<float>& s : trainings_data) {
			train_sample(i, s, context, model);
			i++;
		}

		context.dev_weights.synchFromDevice(model.getWeights());
	}

	neuronalNetwork::test_result_t neuronalNetwork::test(cuda::model& model, std::vector<data::sample<float>>& test_data) {
		data_context context(current_config_, model, test_data);
		context.synchronize(current_config_, model, test_data);

		test_result_t result;
		int i = 0;

		for (data::sample<float>& s : test_data) {
			if (test_sample(i, s, context, model)) {
				result.correct++;
			}

			i++;
		}

		result.total = test_data.size();
		result.error = result.total - result.correct;
		result.ratio = (float) result.correct / (float) result.total;

		return result;
	}

	bool neuronalNetwork::train_sample(const int i, const data::sample<float>& sample, data_context& context, cuda::model& model) {
		int num_blocks;
		int num_threads;

		num_blocks = context.hidden_layer.size();
		num_threads = sample.size();
		cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>((&context.dev_input) + i * sample.size(), &context.dev_hidden, &context.dev_weights);
		cudaThreadSynchronize();

		if (cudaSuccess != cudaGetLastError()) {
			return false;
		}

		num_blocks = context.output_layer.size();
		num_threads = context.hidden_layer.size();
		cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&context.dev_hidden, &context.dev_output, (&context.dev_weights) + ((sample.size() + 1) * context.hidden_layer.size()));
		cudaThreadSynchronize();

		if (cudaSuccess != cudaGetLastError()) {
			return false;
		}

		num_blocks = 1;
		num_threads = context.output_layer.size();
		cuda_neural_network_output_error<<<num_blocks, num_threads>>>(&context.dev_output, (&context.dev_labels) + i * context.output_layer.size());
		cudaThreadSynchronize();

		if (cudaSuccess != cudaGetLastError()) {
			return false;
		}

		num_blocks = context.hidden_layer.size();
		num_threads = context.output_layer.size();
		context.dev_mode.synchToDevice(1);
		cuda_neural_network_error<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&context.dev_hidden, &context.dev_output, (&context.dev_weights) + ((sample.size() + 1) * context.hidden_layer.size()), &context.dev_learning, &context.dev_mode);
		cudaThreadSynchronize();

		if (cudaSuccess != cudaGetLastError()) {
			return false;
		}

		num_blocks = sample.size();
		num_threads = context.hidden_layer.size();
		context.dev_mode.synchToDevice(0);
		cuda_neural_network_error<<<num_blocks, num_threads, num_threads * sizeof(float)>>>((&context.dev_input) + i * sample.size(), &context.dev_hidden, &context.dev_weights, &context.dev_learning, &context.dev_mode);
		cudaThreadSynchronize();

		if (cudaSuccess != cudaGetLastError()) {
			return false;
		}

		return true;
	}

	bool neuronalNetwork::test_sample(const int i, const data::sample<float>& sample, data_context& context, cuda::model& model) {
		int num_blocks;
		int num_threads;

		num_blocks = context.hidden_layer.size();
		num_threads = sample.size();
		cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>((&context.dev_input) + i * sample.size(), &context.dev_hidden, &context.dev_weights);
		cudaThreadSynchronize();

		if (cudaSuccess != cudaGetLastError()) {
			return false;
		}

		num_blocks = context.output_layer.size();
		num_threads = context.hidden_layer.size();
		cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&context.dev_hidden, &context.dev_output, (&context.dev_weights) + ((sample.size() + 1) * context.hidden_layer.size()));
		cudaThreadSynchronize();

		if (cudaSuccess != cudaGetLastError()) {
			return false;
		}

		context.dev_output.synchFromDevice(context.output_layer);
		cudaThreadSynchronize();

		int maxIndex = std::distance(context.output_layer.begin(), std::max_element(context.output_layer.begin(), context.output_layer.end()));

		return (sample.getLabel() == maxIndex);
	}


	neuronalNetwork::data_context::data_context(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples)
	: hidden_layer(config.num_hidden),
	output_layer(config.num_output),
	labels(samples.size() * output_layer.size()),
	dev_input(samples[0].internalData(), samples.size()),
	dev_hidden(hidden_layer),
	dev_output(output_layer),
	dev_weights(model.getWeights()),
	dev_labels(labels)
	{
		for (int i = 0; i < samples.size(); i++) {
			for (int j = 0; j < output_layer.size(); j++) {
				if (samples[i].getLabel() == j) {
					labels[i * output_layer.size() + j] = 1;
				} else {
					labels[i * output_layer.size() + j] = 0;
				}
			}
		}
	}

	void neuronalNetwork::data_context::synchronize(const config_t config, const cuda::model& model, const std::vector<data::sample<float>>& samples) {
		for (int i = 0; i < samples.size(); i++) {
			dev_input.synchToDevice(samples[i].internalData(), i);
		}

		dev_hidden.synchToDevice(hidden_layer);
		dev_output.synchToDevice(output_layer);
		dev_weights.synchToDevice(model.getWeights());
		dev_labels.synchToDevice(labels);
		dev_learning.synchToDevice(config.learning_rate);
	}

}
