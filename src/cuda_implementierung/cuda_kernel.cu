__device__ float activation(float input) {
	return 1 / (1 + (exp((float) -1 * (input))));
}

__global__ void cuda_neural_network(float* input, float* next, float* weights) {
	extern __shared__ float buffer[];
	int lenInput = blockDim.x;
	float inputWeight;
	float inputBias;
	float tmp;

	inputWeight = weights[(lenInput + 1) * blockIdx.x + threadIdx.x];

	if (threadIdx.x == 0) {
		inputBias = weights[(lenInput + 1) * blockIdx.x + lenInput];
	}

	buffer[threadIdx.x] = input[threadIdx.x] * inputWeight;

	__syncthreads();
	for (int i = 1; i < lenInput; i *= 2) {
		if ((threadIdx.x + i) < lenInput) {
			tmp = buffer[threadIdx.x + i];
			__syncthreads();
			buffer[threadIdx.x] += tmp;
			__syncthreads();
		}
	}


	if (threadIdx.x == 0) {
		next[blockIdx.x] = activation(buffer[0] + inputBias);
	}
}

__global__ void cuda_neural_network_output_error(float* output, float* labels) {
	float o = output[threadIdx.x];

	output[threadIdx.x] = o * (1 - o) * (labels[threadIdx.x] - o);
}

__global__ void cuda_neural_network_error(float* current, float* next,
		float* weights, float* learning) {
	extern __shared__ float buffer[];
	int lenInput = blockDim.x;
	float hiddenWeight;
	float hiddenBias;
	float error;
	float tmp;

	hiddenWeight = weights[blockIdx.x + threadIdx.x * (gridDim.x + 1)];

	if (blockIdx.x == 0) {
		hiddenBias = weights[(threadIdx.x + 1) * (gridDim.x + 1) - 1];
	}

	error = next[threadIdx.x];
	buffer[threadIdx.x] = error * hiddenWeight;

	__syncthreads();
	for (int i = 1; i < lenInput; i *= 2) {
		if ((threadIdx.x + i) < lenInput) {
			tmp = buffer[threadIdx.x + i];
			__syncthreads();
			buffer[threadIdx.x] += tmp;
			__syncthreads();
		}
	}

	if (threadIdx.x == 0) {
		float output = current[blockIdx.x];

		buffer[1] = output;
		buffer[2] = *learning;

		current[blockIdx.x] = output * (1 - output) * buffer[0];
	}
	__syncthreads();

	hiddenWeight += buffer[1] * error * buffer[2];

	weights[blockIdx.x + threadIdx.x * (gridDim.x + 1)] = hiddenWeight;

	if (blockIdx.x == 0) {
		hiddenBias += error * buffer[2];

		weights[(threadIdx.x + 1) * (gridDim.x + 1) - 1] = hiddenBias;
	}
}
