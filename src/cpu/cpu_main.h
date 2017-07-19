#ifndef cpu_main_h
#define cpu_main_h

#include <vector>
#include "../data/sample_set.h"
#include "neuronal_network.h"

namespace cpu {
bool main(cpu::neuronal_network& NN, std::vector<data::sample<float>>& trainingsData, std::vector<data::sample<float>>& testData, const int useGui, util::config_file& config);

}
#endif
