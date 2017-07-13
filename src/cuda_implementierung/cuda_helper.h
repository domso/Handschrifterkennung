#ifndef cuda_helper_h
#define cuda_helper_h

#include <cuda.h>
#include <vector>

namespace cuda {

    template<typename synchT>
    class ressource {
    public:

        ressource(const int n = 1) {
            m_error = (cudaMalloc((void**) &m_devData, n * sizeof(synchT)) != cudaSuccess);
        }

        ressource(const std::vector<synchT>& input, const int n = 1) {
            m_error = (cudaMalloc((void**) &m_devData, n * input.size() * sizeof(synchT)) != cudaSuccess);
		}

        ~ressource() {
            cudaFree(m_devData);
        }

        ressource(const ressource& o) = delete;
        ressource(const ressource&& o) = delete;

        synchT* operator &() {
        	return m_devData;
        }

        bool has_error() const {
            return m_error;
        }

        void synch_to_device(const synchT& input, const int n = 1) {
            m_error = (cudaMemcpy(m_devData, &input, n * sizeof(synchT), cudaMemcpyHostToDevice) != cudaSuccess);
        }

        void synch_from_device(synchT& output, const int n = 1) {
            m_error = (cudaMemcpy(&output, m_devData, n * sizeof(synchT), cudaMemcpyDeviceToHost) != cudaSuccess);
        }

        void synch_to_device(const std::vector<synchT>& input, const int offset = 0) {
            m_error = (cudaMemcpy(m_devData + offset * input.size(), input.data(), input.size() * sizeof(synchT), cudaMemcpyHostToDevice) != cudaSuccess);
        }

        void synch_from_device(std::vector<synchT>& output, const int offset = 0) {
            m_error = (cudaMemcpy(output.data(), m_devData + offset * output.size(), output.size() * sizeof(synchT), cudaMemcpyDeviceToHost) != cudaSuccess);
        }

    private:
        synchT* m_devData;
        bool m_error;
    };

}

#endif
