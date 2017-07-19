#ifndef cuda_ressource_h
#define cuda_ressource_h

#include <cuda.h>
#include <vector>

namespace cuda {
/*
 * RAII-wrapper for cudaMalloc and cudaMemcpy
 */
template<typename synchT>
class ressource {
public:

	/*
	 * Allocates memory for n instances of synchT (n * sizeof(synchT) Bytes) on the graphic-card
	 * (see cudaMalloc)
	 * @param n: optional number of requested instances
	 */
	ressource(const int n = 1) {
		m_error = (cudaMalloc((void**) &m_devData, n * sizeof(synchT))
				!= cudaSuccess);
		m_size = n * sizeof(synchT) * m_error;
	}

	/*
	 * Allocates memory for n chunks on the graphic-card.
	 * (n * input.size() * sizeof(synch) Bytes)
	 * (see cudaMalloc)
	 * @param input: vector containing the data of a single chunk
	 * @param n: optional number of requested chunks
	 */
	ressource(const std::vector<synchT>& input, const int n = 1) {
		m_error = (cudaMalloc((void**) &m_devData,
				n * input.size() * sizeof(synchT)) != cudaSuccess);
		m_size = n * input.size() * sizeof(synchT) * m_error;
	}

	/*
	 * Frees the allocated memory
	 */
	~ressource() {
		cudaFree(m_devData);
	}

	ressource(const ressource& o) = delete;
	ressource(const ressource&& o) = delete;

	/*
	 * @return: device_pointer for the cuda-kernels
	 */
	synchT* operator &() {
		return m_devData;
	}

	/*
	 * @return: true, if the all operations could be executed successfully
	 */
	bool has_error() const {
		return m_error;
	}

	/*
	 * Copies n instances of synchT to the graphic-card.
	 * (see cudaMemcpy)
	 * @param input: reference to n instances of synchT
	 * @param n: number of instances at input
	 */
	void synch_to_device(const synchT& input, const int n = 1) {
		m_error &= (m_size <= n * sizeof(synchT))
				&& (cudaMemcpy(m_devData, &input, n * sizeof(synchT),
						cudaMemcpyHostToDevice) != cudaSuccess);
	}

	/*
	 * Copies n instances of synchT to the RAM.
	 * (see cudaMemcpy)
	 * @param output: reference to n instances of synchT
	 * @param n: number of instances at output
	 */
	void synch_from_device(synchT& output, const int n = 1) {
		m_error &= (m_size <= n * sizeof(synchT))
				&& (cudaMemcpy(&output, m_devData, n * sizeof(synchT),
						cudaMemcpyDeviceToHost) != cudaSuccess);
	}

	/*
	 * Copies a vector of synchT-instances to the graphic-card.
	 * (see cudaMemcpy)
	 * @param input: input data
	 * @param offset: start-offset in the memory
	 */
	void synch_to_device(const std::vector<synchT>& input,
			const int offset = 0) {
		m_error &= (m_size <= offset * input.size() * sizeof(synchT) + input.size() * sizeof(synchT))
				&& (cudaMemcpy(m_devData + offset * input.size(), input.data(),
						input.size() * sizeof(synchT), cudaMemcpyHostToDevice)
						!= cudaSuccess);
	}

	/*
	 * Copies a vector of synchT-instances to the RAM
	 * (see cudaMemcpy)
	 * @param output: output data
	 * @param offset: start-offset in the memory
	 */
	void synch_from_device(std::vector<synchT>& output, const int offset = 0) {
		m_error &= (m_size <= offset * output.size() * sizeof(synchT) + output.size() * sizeof(synchT))
				&& (cudaMemcpy(output.data(),
						m_devData + offset * output.size(),
						output.size() * sizeof(synchT), cudaMemcpyDeviceToHost)
						!= cudaSuccess);
	}

private:
	int m_size;
	synchT* m_devData;
	bool m_error;
};

}

#endif
