__device__ float activation(float input) {
	return 1 / (1 + (exp((float) -1 * (input))));
}

__global__ void cuda_neural_network(float* input, float* next, float* weights) {
	extern __shared__ float buffer[];
	int len_input = blockDim.x;
	float input_weight;
	float input_bias;

	input_weight = weights[(len_input + 1) * blockIdx.x + threadIdx.x];

	if (threadIdx.x == 0) {
		input_bias = weights[(len_input + 1) * blockIdx.x + len_input];
	}

	buffer[threadIdx.x] = input[threadIdx.x] * input_weight;

	__syncthreads();

	for (int i = 1; i < len_input; i *= 2) {
		__syncthreads();
		if ((threadIdx.x + i) < len_input) {
			__syncthreads();
			buffer[threadIdx.x] += buffer[threadIdx.x + i];
			__syncthreads();
		}
		__syncthreads();
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		next[blockIdx.x] = activation(buffer[0] + input_bias);
	}
}

__global__ void cuda_neural_network_output_error(float* output, float* labels) {
	float o = output[threadIdx.x];

	output[threadIdx.x] = o * (1 - o) * (labels[threadIdx.x] - o);
}

__global__ void cuda_neural_network_error(float* current, float* next,
		float* weights, float* learning, int* mode) {
	extern __shared__ float buffer[];
	int len_input = blockDim.x;
	float hidden_weight;
	float hidden_bias;
	float error;

	hidden_weight = weights[blockIdx.x + threadIdx.x * (gridDim.x + 1)];

	if (blockIdx.x == 0) {
		hidden_bias = weights[(threadIdx.x + 1) * (gridDim.x + 1) - 1];
	}

	error = next[threadIdx.x];
	buffer[threadIdx.x] = error * hidden_weight;

	__syncthreads();

	for (int i = 1; i < len_input; i *= 2) {
		__syncthreads();
		if ((threadIdx.x + i) < len_input) {
			__syncthreads();
			buffer[threadIdx.x] += buffer[threadIdx.x + i];
			__syncthreads();
		}
		__syncthreads();
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		float output = current[blockIdx.x];

		buffer[1] = output;
		buffer[2] = *learning;

		if (*mode != 0) {
			current[blockIdx.x] = output * (1 - output) * buffer[0];
		}
	}
	__syncthreads();

	hidden_weight += buffer[1] * error * buffer[2];

	weights[blockIdx.x + threadIdx.x * (gridDim.x + 1)] = hidden_weight;

	if (blockIdx.x == 0) {
		hidden_bias += error * buffer[2];

		weights[(threadIdx.x + 1) * (gridDim.x + 1) - 1] = hidden_bias;
	}
}
