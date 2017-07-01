#include <iostream>
#include "src/sample_set.h"
#include "src/cuda_helper.h"
#include "src/cuda_kernel.h"


int main(int argc, char** argv) {
    auto input = data::sample_set::load<float>("../train-images.idx3-ubyte", "../train-labels.idx1-ubyte");

	std::vector<float>& ref_input = input[0].internalData();

    std::vector<float> hidden(20);
    int output_size = 10;
    std::vector<float> output(input.size() * output_size);
    std::vector<float> weights((ref_input.size() + 1) * hidden.size() + (hidden.size() + 1) * output_size);
    std::vector<int> lengths(3);
    int n = input.size();

    lengths[0] = ref_input.size();
    lengths[1] = hidden.size();
    lengths[2] = output_size;

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

    cuda_helper::ressource<int> dev_lengths(lengths);
    dev_lengths.synchToDevice(lengths);

    cuda_helper::ressource<int> dev_n(1);
    dev_n.synchToDevice(n, 1);

    int num_blocks = std::max<int>(lengths[1], lengths[2]);
    int num_threads = std::max<int>(lengths[0], lengths[1]);

    std::cout << "num_blocks = " << num_blocks << std::endl;
    std::cout << "num_threads = " << num_threads << std::endl;

    cuda_neural_network<<<num_blocks, num_threads, num_threads * sizeof(float)>>>(&dev_input, &dev_hidden, &dev_output, &dev_lengths, &dev_n, &dev_weights);

    cudaThreadSynchronize();

    if (cudaSuccess != cudaGetLastError()) {
    	std::cout << "cuda_neural_network error" << std::endl;
	}

    std::cout << "finished" << std::endl;

    return 0;
}
