#ifndef cuda_main_h
#define cuda_main_h

#include <vector>
#include "../data/sample_set.h"
#include "cuda_model.h"
#include "cuda_neuronal_network.h"

namespace cuda {
bool main(cuda::neuronal_network& NN, std::vector<data::sample<float>>& trainingsData, std::vector<data::sample<float>>& testData, const int useGui, util::config_file& config);

}
#endif
