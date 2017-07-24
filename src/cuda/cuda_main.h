#ifndef cuda_main_h
#define cuda_main_h

#include <vector>
#include "../data/sample_set.h"
#include "cuda_model.h"
#include "cuda_neuronal_network.h"

namespace cuda {
/*
 * Cuda-main implementation
 * Trains a new model on the trainingsData and tests it on the testData.
 * Additional parameters are stored in the given config.
 * If the useGui-flag was set, the NN will be prepared for additional NN::classify() calls. (for the GUI)
 * @param NN: initialized neuronal-network
 * @param trainingsData: non empty vector containing the labeled trainings-samples
 * @param testData: non empty vector containing the labeled test-samples
 * @param useGui: gui-window-flag
 * @param config: initialized config
 * @return: true on success
 */
bool main(cuda::neuronal_network& NN, const std::vector<data::sample<float>>& trainingsData, const std::vector<data::sample<float>>& testData, const int useGui, const util::config_file& config);

}
#endif
