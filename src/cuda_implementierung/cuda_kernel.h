#ifndef cuda_kernel_h
#define cuda_kernel_h


__global__ void cuda_neural_network(float* input, float* next, float* weights);

__global__ void cuda_neural_network_output_error(float* output, float* labels);

__global__ void cuda_neural_network_error(float* current, float* next, float* weights, float* learning, int* mode);


#endif
