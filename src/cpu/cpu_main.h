#ifndef cpu_main_h
#define cpu_main_h

#include <vector>
#include "../data/sample_set.h"
#include "neuronal_network.h"

namespace cpu {
/**
 * trains and tests the given network with the given data
 * @param NN  the network to train and test
 * @param trainingsData  the data used for training
 * @param testData  the data used for testing
 * @param useGui  flag, if the gui should be showed
 * @param config  the config-file to use
 * @return: true on success
 */
bool main(cpu::neuronal_network& NN, const std::vector<data::sample<float>>& trainingsData, const std::vector<data::sample<float>>& testData, const int useGui, const util::config_file& config);

}
#endif
