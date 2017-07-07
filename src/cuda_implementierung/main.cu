
#include <iostream>
#include "../sample_set.h"
#include "cuda_neuronalNetwork.h"
#include "cuda_model.h"

int main3() {
	auto trainings_data = data::sample_set::load<float>("./train-images.idx3-ubyte",
				"./train-labels.idx1-ubyte");
	auto test_data = data::sample_set::load<float>("./t10k-images.idx3-ubyte",
					"./t10k-labels.idx1-ubyte");

	cuda::model model;
	cuda::neuronalNetwork NN;

	NN.train(model, trainings_data);

	auto result = NN.test(model, test_data);

	std::cout << result.correct << std::endl;
	std::cout << result.total << std::endl;
	std::cout << result.ratio << std::endl;

	return 0;
}

/*#include <iostream>
#include "src/sample_set.h"
#include "src/cuda_helper.h"
#include "src/cuda_kernel.h"
#include <random>

int main_old(int argc, char** argv) {
	auto input = data::sample_set::load<float>("../train-images.idx3-ubyte",
			"../train-labels.idx1-ubyte");

	input[1] = input[0];
	std::vector<float>& ref_input = input[0].internalData();

	std::vector<float> hidden(20);
	std::vector<float> output(10);
	std::vector<float> weights(
			(ref_input.size() + 1) * hidden.size()
					+ (hidden.size() + 1) * output.size());
	std::vector<float> labels(input.size() * output.size());
	float learning_rate = 0.15;

	for (int i = 0; i < input.size(); i++) {
		for (int j = 0; j < output.size(); j++) {
			if (input[i].getLabel() == j) {
				labels[i * output.size() + j] = 1;
			} else {
				labels[i * output.size() + j] = 0;
			}
		}
	}
	std::srand(0);

	for (float& x : weights) {
		x = -0.5 + (float) (std::rand() % 1000001) / (float) 1000000;
		//std::rand() / (float)RAND_MAX;
	}

	cuda_helper::ressource<float> dev_input(ref_input, input.size());

	for (int i = 0; i < input.size(); i++) {
		dev_input.synchToDevice(input[i].internalData(), i);
	}

	cuda_helper::ressource<float> dev_hidden(hidden);
	dev_hidden.synchToDevice(hidden);

	cuda_helper::ressource<float> dev_output(output);
	dev_output.synchToDevice(output);

	cuda_helper::ressource<float> dev_weights(weights);
	dev_weights.synchToDevice(weights);

	cuda_helper::ressource<float> dev_labels(labels);
	dev_labels.synchToDevice(labels);

	cuda_helper::ressource<int> dev_mode;

	cuda_helper::ressource<float> dev_learning;
	dev_learning.synchToDevice(learning_rate);

	int num_blocks;
	int num_threads;
	int num_samples = 60000;
	int correct = 0;
	float max = 0;
	int maxIndex;

	for (int iteration = 0; iteration < 1; iteration++) {
		correct = 0;
		for (int i = 0; i < num_samples; i++) {
			num_blocks = hidden.size();
			num_threads = ref_input.size();
			cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>((&dev_input) + i * ref_input.size(), &dev_hidden, &dev_weights);
			cudaThreadSynchronize();

			num_blocks = output.size();
			num_threads = hidden.size();
			cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&dev_hidden, &dev_output, (&dev_weights) + ((ref_input.size() + 1) * hidden.size()));
			cudaThreadSynchronize();

			dev_output.synchFromDevice(output);
			cudaThreadSynchronize();

			maxIndex = -1;
			max = 0;
			for (int m = 0; m < output.size(); m++) {
				if (output[m] >= max) {
					max = output[m];
					maxIndex = m;
				}
				//std::cout << output[m] << std::endl;
			}
			//std::cout << (int)input[i].getLabel() << ": " << maxIndex << std::endl;
			if (input[i].getLabel() == maxIndex && maxIndex >= 0) {
				correct++;
			}

			num_blocks = 1;
			num_threads = output.size();
			cuda_neural_network_output_error<<<num_blocks, num_threads>>>(&dev_output, (&dev_labels) + i * output.size());
			cudaThreadSynchronize();

			num_blocks = hidden.size();
			num_threads = output.size();
			dev_mode.synchToDevice(1);
			cuda_neural_network_error<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&dev_hidden, &dev_output, (&dev_weights) + ((ref_input.size() + 1) * hidden.size()), &dev_learning, &dev_mode);
			cudaThreadSynchronize();

			num_blocks = ref_input.size();
			num_threads = hidden.size();
			dev_mode.synchToDevice(0);
			cuda_neural_network_error<<<num_blocks, num_threads, num_threads * sizeof(float)>>>((&dev_input) + i * ref_input.size(), &dev_hidden, &dev_weights, &dev_learning, &dev_mode);
			cudaThreadSynchronize();

			if (cudaSuccess != cudaGetLastError()) {
				std::cout << "cuda_neural_network error" << std::endl;
			}
		}

		std::cout << correct << " / " << input.size() << " correct"
				<< std::endl;
	}

	auto inputTest = data::sample_set::load<float>("../t10k-images.idx3-ubyte",
			"../t10k-labels.idx1-ubyte");

	std::vector<float> ref_inputTest = inputTest[0].internalData();
	cuda_helper::ressource<float> dev_inputTest(inputTest[0].internalData(),
			inputTest.size());

	for (int i = 0; i < inputTest.size(); i++) {
		dev_inputTest.synchToDevice(inputTest[i].internalData(), i);
	}

	correct = 0;
	for (int i = 0; i < inputTest.size(); i++) {
		num_blocks = hidden.size();
		num_threads = ref_inputTest.size();
		cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>((&dev_inputTest) + i * ref_inputTest.size(), &dev_hidden, &dev_weights);
		cudaThreadSynchronize();

		num_blocks = output.size();
		num_threads = hidden.size();
		cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&dev_hidden, &dev_output, (&dev_weights) + ((ref_inputTest.size() + 1) * hidden.size()));
		cudaThreadSynchronize();

		dev_output.synchFromDevice(output);
		cudaThreadSynchronize();

		maxIndex = -1;
		max = 0;
		for (int m = 0; m < output.size(); m++) {
			if (output[m] >= max) {
				max = output[m];
				maxIndex = m;
			}
			//std::cout << output[m] << std::endl;
		}
		//std::cout << (int)input[i].getLabel() << ": " << maxIndex << std::endl;
		if (inputTest[i].getLabel() == maxIndex && maxIndex >= 0) {
			correct++;
		}
	}

	std::cout << correct << " / " << inputTest.size() << " correct" << std::endl;

	return 0;
}
*/
/*
 input[1] = input[0];
 std::vector<float>& ref_input = input[0].internalData();

 std::vector<float> hidden(20, 7);
 int output_size = 10;
 std::vector<float> output(input.size() * output_size);
 std::vector<float> weights(
 (ref_input.size() + 1) * hidden.size()
 + (hidden.size() + 1) * output_size);
 std::vector<int> lengths(3);
 std::vector<float> labels(input.size() * output_size);
 int n = 2;//input.size();
 float learning_rate = 0.2;

 for (int i = 0; i < input.size(); i++) {
 for (int j = 0; j < output_size; j++) {
 if (input[i].getLabel() == j) {
 labels[i * output_size + j] = 1;
 } else {
 labels[i * output_size + j] = 0;
 }
 }
 }

 std::srand(0);

 for (float& x : weights) {
 x = std::rand() / (float)RAND_MAX;
 }

 std::cout << weights.size() << std::endl;

 lengths[0] = ref_input.size();
 lengths[1] = hidden.size();
 lengths[2] = output_size;

 cuda_helper::ressource<float> dev_input(ref_input, input.size());

 for (int i = 0; i < input.size(); i++) {
 dev_input.synchToDevice(input[i].internalData(), i);
 }

 cuda_helper::ressource<float> dev_labels(labels);
 dev_labels.synchToDevice(labels);

 cuda_helper::ressource<float> dev_hidden(hidden);
 dev_hidden.synchToDevice(hidden);

 cuda_helper::ressource<float> dev_output(output);
 dev_output.synchToDevice(output);

 cuda_helper::ressource<float> dev_weights(weights);
 dev_weights.synchToDevice(weights);

 cuda_helper::ressource<int> dev_lengths(lengths);
 dev_lengths.synchToDevice(lengths);

 cuda_helper::ressource<int> dev_n;
 dev_n.synchToDevice(n);

 cuda_helper::ressource<float> dev_learning;
 dev_learning.synchToDevice(learning_rate);

 int num_blocks = std::max<int>(lengths[1], lengths[2]);
 int num_threads = std::max<int>(lengths[0], lengths[1]);

 std::cout << "num_blocks = " << num_blocks << std::endl;
 std::cout << "num_threads = " << num_threads << std::endl;

 cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&dev_input, &dev_hidden, &dev_output, &dev_lengths, &dev_n, &dev_weights, &dev_labels, &dev_learning);

 cudaThreadSynchronize();

 if (cudaSuccess != cudaGetLastError()) {
 std::cout << "cuda_neural_network error" << std::endl;
 }

 dev_output.synchFromDevice(output);

 cudaThreadSynchronize();

 if (cudaSuccess != cudaGetLastError()) {
 std::cout << "dev_weights synchFromDevice error" << std::endl;
 }

 int correct = 0;
 float max = 0;
 int maxIndex;

 for (int i = 0; i < n; i++) {
 maxIndex = -1;
 for (int j = 0; j < output_size; j++) {
 if (output[i * output_size + j] >= max) {
 max = output[i * output_size + j];
 maxIndex = j;

 std::cout << output[i * output_size + j] << std::endl;
 } else {
 //if (output[i * output_size + j] != 0) {
 std::cout << output[i * output_size + j] << std::endl;
 //}
 }
 }
 std::cout << "..." << std::endl;
 if (labels[i] == maxIndex && maxIndex >= 0) {
 correct++;
 }
 }
 std::cout << correct << " / " << input.size() << " correct" << std::endl;


 std::cout << "finished" << std::endl;
 */

