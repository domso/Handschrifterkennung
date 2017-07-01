__global__ void cuda_neural_network(float* input, float* hidden, float* output,
		int* lengths, int* n, float* weights) {

	extern __shared__ float buffer[];

	int len_input = lengths[0];
	int len_hidden = lengths[1];
	int len_output = lengths[2];

	float input_weight = 0;
	if (blockIdx.x < len_hidden && threadIdx.x < len_input) {
		input_weight = weights[(len_input + 1) * blockIdx.x + threadIdx.x];
	}

	float input_bias;
	if (threadIdx.x == 0) {
		input_bias = weights[blockDim.x];
	}

	float hidden_weight;
	if (blockIdx.x < len_output && threadIdx.x < len_hidden) {
		hidden_weight = weights[(len_input + 1) * len_hidden + (len_hidden + 1) * blockIdx.x + threadIdx.x];
	}

	float hidden_bias;
	if (threadIdx.x == 0) {
		input_bias = weights[(len_input + 1) * (blockIdx.x + 1) + blockDim.x];
	}

	int end = len_input * (*n);
	int output_offset = 0;
	for (int offset = 0; offset < end; offset += len_input) {
		//-----------------------input-----------------------
		if (threadIdx.x < len_input && blockIdx.x < len_hidden) {
			buffer[threadIdx.x] = input[offset + threadIdx.x] * input_weight;

			for (int i = 1; i < len_input; i *= 2) {
				if (threadIdx.x + i < len_input) {
					buffer[threadIdx.x] += buffer[threadIdx.x + i];
				}

			}

			if (threadIdx.x == 0) {
				hidden[blockIdx.x] = buffer[0] + input_bias;
			}
		}
		__syncthreads();
		//-----------------------hidden-----------------------
		if (threadIdx.x < len_hidden && blockIdx.x < len_output) {
			buffer[threadIdx.x] = hidden[threadIdx.x] * hidden_weight;

			for (int i = 1; i < len_hidden; i *= 2) {
				if (threadIdx.x + i < len_hidden) {
					buffer[threadIdx.x] += buffer[threadIdx.x + i];
				}
			}

			if (threadIdx.x == 0) {
				output[output_offset + blockIdx.x] = buffer[0] + hidden_bias;
				output_offset += len_output;
			}
		}
		__syncthreads();
	}

}
