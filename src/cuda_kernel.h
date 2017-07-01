#ifndef cuda_kernel_h
#define cuda_kernel_h

__global__ void cuda_neural_network(float* input, float* hidden, float* output, int* lengths, int* n, float* weights);


#endif
