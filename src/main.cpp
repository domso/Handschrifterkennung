#include <iostream>
#include "config_file.h"

void execute_cpp(util::config_file& config) {

}

void execute_cuda(util::config_file& config) {

}

int main(int argc, char* args[]) {

	if (argc < 2) {
		std::cout << "[Error] No arguments" << std::endl;

		return -1;
	}

	std::string filename(args[1]);
	util::config_file config;

	config.load(filename);

	if (config.isError()) {
		std::cout << config.getErrorMsgs() << std::endl;

		return -1;
	}

	config.requireString("implementation");
	config.requireString("dataset-training-samples");
	config.requireString("dataset-training-labels");
	config.requireString("dataset-testing-samples");
	config.requireString("dataset-testing-labels");
	config.recommendNumeric<int>("use-gui", 1);
	config.recommendNumeric<int>("num-hidden", 20);
	config.recommendNumeric<int>("num-relearning", 1);
	config.recommendNumeric<float>("learning-rate", 0.2);

	if (config.isError()) {
		std::cout << config.getErrorMsgs() << std::endl;

		return -1;
	}

	if (config.isWarning()) {
		std::cout << config.getWarningMsgs() << std::endl;
	}

	std::string impl = config.getString("implementation");
	if (impl == "c++") {
		execute_cpp(config);
	} else if (impl == "cuda") {
		execute_cuda(config);
	} else {
		std::cout << "[ERROR] Unsupported implementation: " << impl
				<< std::endl;
		return -1;
	}

	return 0;
}
