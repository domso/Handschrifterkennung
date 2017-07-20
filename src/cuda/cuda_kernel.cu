__device__ float activation(float input) {
	return 1 / (1 + (exp((float) -1 * (input))));
}

__global__ void cuda_neural_network(float* input, float* next, float* weights) {
	extern __shared__ float buffer[];
	float inputWeight;
	float inputBias;
	float tmp;

	inputWeight = weights[(blockDim.x + 1) * blockIdx.x + threadIdx.x];

	if (threadIdx.x == 0) {
		inputBias = weights[(blockDim.x + 1) * blockIdx.x + blockDim.x];
	}

	tmp = input[threadIdx.x] * inputWeight;
	buffer[threadIdx.x] = tmp;

	__syncthreads();
#pragma unroll
	for (int i = 1; i < blockDim.x; i *= 2) {
		int j = threadIdx.x + i;
		if (j < blockDim.x) {
			tmp += buffer[j];
			__syncthreads();
			buffer[threadIdx.x] = tmp;
			__syncthreads();
		}
	}

	if (threadIdx.x == 0) {
		next[blockIdx.x] = activation(tmp + inputBias);
	}
}

__global__ void cuda_neural_network_error(float* current, float* next,
		float* weights, float* learning, float* labels) {
	extern __shared__ float buffer[];
	float weight;
	float bias;
	float error;
	float tmp;
	float output;
	float l;

	int weightIndex = blockIdx.x + threadIdx.x * (gridDim.x + 1);
	weight = weights[weightIndex];

	if (blockIdx.x == 0) {
		bias = weights[(threadIdx.x + 1) * (gridDim.x + 1) - 1];
	}

	error = next[threadIdx.x];
	if (labels != NULL) {
		error = error * (1 - error) * (labels[threadIdx.x] - error);
	}

	tmp = error * weight;
	buffer[threadIdx.x] = tmp;
	__syncthreads();

#pragma unroll
	for (int i = 1; i < blockDim.x; i *= 2) {
		int j = threadIdx.x + i;
		if (j < blockDim.x) {
			tmp += buffer[j];
			__syncthreads();
			buffer[threadIdx.x] = tmp;
			__syncthreads();
		}
	}

	if (threadIdx.x == 0) {
		output = current[blockIdx.x];
		l = *learning;
		buffer[1] = output * l;
		buffer[2] = l;
	}
	__syncthreads();

	weights[weightIndex] = weight + buffer[1] * error;

	if (blockIdx.x == 0) {
		weights[(threadIdx.x + 1) * (gridDim.x + 1) - 1] = bias + error * buffer[2];
	}

	if (threadIdx.x == 0) {
		current[blockIdx.x] = output * (1 - output) * tmp;
	}
}
