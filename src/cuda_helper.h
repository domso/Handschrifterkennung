#ifndef cuda_helper_h
#define cuda_helper_h

#include <cuda.h>
#include <vector>

namespace cuda_helper {

    template<typename synchT>
    class ressource {
    public:

        ressource(const int n = 1) {
            error_ = (cudaMalloc((void**) &dev_data_, n * sizeof(synchT)) != cudaSuccess);
        }

        ressource(const std::vector<synchT>& input, const int n = 1) {
                    error_ = (cudaMalloc((void**) &dev_data_, n * input.size() * sizeof(synchT)) != cudaSuccess);
		}

        ~ressource() {
            cudaFree(dev_data_);
        }

        ressource(const ressource& o) = delete;
        ressource(const ressource&& o) = delete;

        synchT* operator &() {
        	return dev_data_;
        }

        bool hasError() const {
            return error_;
        }

        void synchToDevice(const synchT& input, const int n = 1) {
            error_ = (cudaMemcpy(dev_data_, &input, n * sizeof(synchT), cudaMemcpyHostToDevice) != cudaSuccess);
        }


        void synchFromDevice(synchT& output, const int n = 1) {
            error_ = (cudaMemcpy(&output, dev_data_, n * sizeof(synchT), cudaMemcpyDeviceToHost) != cudaSuccess);
        }

        void synchToDevice(std::vector<synchT>& input, const int offset = 0) {
            error_ = (cudaMemcpy(dev_data_ + offset * input.size(), input.data(), input.size() * sizeof(synchT), cudaMemcpyHostToDevice) != cudaSuccess);
        }

        void synchFromDevice(std::vector<synchT>& output, const int offset = 0) {
            error_ = (cudaMemcpy(output.data(), dev_data_ + offset * output.size(), output.size() * sizeof(synchT), cudaMemcpyDeviceToHost) != cudaSuccess);
        }

    private:
        synchT* dev_data_;
        bool error_;
    };

}

#endif
