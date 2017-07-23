#include <iostream>
#include <thread>
#include "util/config_file.h"
#include "parameters.h"
#include "cuda/cuda_main.h"
#include "cuda/cuda_model.h"
#include "cuda/cuda_neuronal_network.h"
#include "gui/basic_interface.h"
#include "cpu/neuronal_network.h"
#include "cpu/cpu_main.h"

/*
 * Opens the gui-window and executes the main update-methode
 * Runs until the window was closed *
 * @param window: non initialized gui-window
 */
void gui_thread(gui::basic_interface& window) {
	window.init();
	window.update();
	window.close();
}

/*
 * Start the gui-thread and waits until the window was closed
 * Every drawn sample on the gui will be classified with
 * NNtype::classify(data::sample<float>&)
 * @param NN: initialized neuronal-network of type NNtype
 * @param window: non initialized gui-window
 * @param sampleWidth: x-resolution of a sample
 * @param sampleHeight: y-resolution of a sample
 */
template<typename NNtype>
void gui_main(NNtype& NN, gui::basic_interface& window, const int sampleWidth, const int sampleHeight) {
	data::sample<float> output(sampleWidth, sampleHeight);
	data::sample<float> final(sampleWidth, sampleHeight);
	std::thread t1(&gui_thread, std::ref(window));

	while (window.is_active()) {
		if (window.wait_for_output(output)) {
			final.normalize_from(output);
			std::cout << (int) NN.classify(final) << std::endl;
		}
	}

	t1.join();
}

/*
 * Initializes the gui and start the gui-main-function
 * Requires NNtype::classify(data::sample<float>&)
 * @param NN: initialized neuronal-network of type NNtype
 * @param trainingsData: non empty vector with trainings-samples as a size-reference
 */
template<typename NNtype>
void gui_init(NNtype& NN, util::config_file& config, std::vector<data::sample<float>>& trainingsData) {
	int sampleWidth = trainingsData[0].get_width();
	int sampleHeight = trainingsData[0].get_height();
	int windowWidth = config.getNumeric<int, parameters::window_width>();
	int windowHeight = config.getNumeric<int, parameters::window_height>();

	gui::basic_interface window(windowWidth, windowHeight, sampleWidth, sampleHeight);

	gui_main<NNtype>(NN, window, sampleWidth, sampleHeight);
}

/*
 * Loads all samples and labels from the files specified in config and sets the useGui-flag
 * If the useGui-flag was set, the gui will be opened afterwards
 * @param config: input-config file
 * @param trainingsData: empty vector for trainings-samples
 * @param testData: empty vector for the test-samples
 * @param useGui: output-flag for gui
 * @return: true on success
 */
bool load_samples(util::config_file& config, std::vector<data::sample<float>>& trainingsData, std::vector<data::sample<float>>& testData, int& useGui) {
	auto pathTrainingSamples = config.getString<parameters::path_training_samples>();
	auto pathTrainingLabels = config.getString<parameters::path_training_labels>();
	auto pathTestingSamples = config.getString<parameters::path_testing_samples>();
	auto pathTestingLabels = config.getString<parameters::path_testing_labels>();

	useGui = config.getNumeric<int, parameters::use_gui>();

	trainingsData = data::sample_set::load<float>(pathTrainingSamples, pathTrainingLabels);
	testData = data::sample_set::load<float>(pathTestingSamples, pathTestingLabels);

	if (trainingsData.size() == 0) {
		std::cout << "[ERROR] Could not load any training-sample!" << std::endl;
		return false;
	}

	if (testData.size() == 0) {
		std::cout << "[ERROR] Could not load any test-sample!" << std::endl;
		return false;
	}

	return true;
}

/*
 * Starts the c++ implementation
 * If the useGui-flag was set, the gui will be opened afterwards
 * @param config: input-config file
 * @param trainingsData: vector with the loaded trainings-samples
 * @param testData: vector with the loaded test-samples
 * @param useGui: flag for gui
 */
void execute_cpu(util::config_file& config, std::vector<data::sample<float>>& trainingsData, std::vector<data::sample<float>>& testData, int& useGui) {
	auto numHidden = config.getNumeric<int, parameters::num_hidden>();
	cpu::neuronal_network NN(trainingsData[0].size(), numHidden, 10);
	cpu::main(NN, trainingsData, testData, useGui, config);

	if (useGui) {
		gui_init<cpu::neuronal_network>(NN, config, trainingsData);
	}
}

/*
 * Starts the cuda implementation
 * If the useGui-flag was set, the gui will be opened afterwards
 * @param config: input-config file
 * @param trainingsData: vector with the loaded trainings-samples
 * @param testData: vector with the loaded test-samples
 * @param useGui: flag for gui
 */
void execute_cuda(util::config_file& config, std::vector<data::sample<float>>& trainingsData, std::vector<data::sample<float>>& testData, int& useGui) {
	cuda::neuronal_network NN;
	cuda::main(NN, trainingsData, testData, useGui, config);

	if (useGui) {
		gui_init<cuda::neuronal_network>(NN, config, trainingsData);
	}
}

/*
 * Implementation switch for cuda and c++
 * Loads the samples using the config-files
 * @param config: input-config file
 */
void execute_general(util::config_file& config) {
	std::vector<data::sample<float>> trainingsData;
	std::vector<data::sample<float>> testData;
	int useGui;

	if (!load_samples(config, trainingsData, testData, useGui)) {
		return;
	}

	std::string implementation = config.getString<parameters::implementation>();
	if (implementation == "c++") {
		execute_cpu(config, trainingsData, testData, useGui);
	} else if (implementation == "cuda") {
		execute_cuda(config, trainingsData, testData, useGui);
	} else {
		std::cout << "[ERROR] Unsupported implementation: " << implementation << std::endl;
	}
}

/*
 * Checks the given config for required and recommended parameters.
 * @param config: input-config file
 */
void check_config(util::config_file& config) {
	config.requireString<parameters::implementation>();
	config.requireString<parameters::path_training_samples>();
	config.requireString<parameters::path_training_labels>();
	config.requireString<parameters::path_testing_samples>();
	config.requireString<parameters::path_testing_labels>();

	config.recommendNumeric<int, parameters::use_gui>();
	config.recommendNumeric<int, parameters::num_hidden>();
	config.recommendNumeric<int, parameters::num_relearning>();
	config.recommendNumeric<float, parameters::learning_rate>();
}

/*
 * Tries to load a configuration from the given file and checks
 * it for any warnings or errors
 * @param config: input-config file
 * @param filename: path to the file
 * @return: false on error
 */
bool load_config(util::config_file& config, const std::string& filename) {
	config.load(filename);

	if (config.isError()) {
		std::cout << config.getErrorMsgs() << std::endl;
		return -1;
	}

	check_config(config);

	if (config.isError()) {
		std::cout << config.getErrorMsgs() << std::endl;
		return -1;
	}

	if (config.isWarning()) {
		std::cout << config.getWarningMsgs() << std::endl;
	}

	return true;
}

/*
 * Loads the config-file specified by the first argument and executes
 * the starts the implementation-switch
 */
int main(int argc, char* args[]) {
	if (argc < 2) {
		std::cout << "[Error] No arguments" << std::endl;
		return -1;
	}

	std::string filename(args[1]);
	util::config_file config;

	if (!load_config(config, filename)) {
		return -1;
	}

	execute_general(config);

	return 0;
}
