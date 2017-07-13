#include <iostream>
#include "../sample_set.h"
#include "cuda_neuronal_network.h"
#include "cuda_model.h"
#include "../basic_interface.h"
#include <thread>
#include <chrono>

void gui_thread(basic_interface& i) {
	i.init();
	i.update();
	i.close();
}

int main() {
	std::mutex m;
	data::sample<float> output(0, 28, 28);
	data::sample<float> final(0, 28, 28);
	basic_interface i(800, 800, 28, 28);

	auto trainingsData = data::sample_set::load<float>(
			"./train-images.idx3-ubyte", "./train-labels.idx1-ubyte");
	auto testData = data::sample_set::load<float>("./t10k-images.idx3-ubyte",
			"./t10k-labels.idx1-ubyte");

	cuda::model model;
	cuda::neuronal_network NN;

	auto tp1 = std::chrono::high_resolution_clock::now();
	NN.train(model, trainingsData);
	auto tp2 = std::chrono::high_resolution_clock::now();

	auto duration  = std::chrono::duration_cast<std::chrono::microseconds>(tp2 - tp1).count();

	std::cout << duration << std::endl;

	auto result = NN.test(model, testData);

	std::cout << result.correct << std::endl;
	std::cout << result.total << std::endl;
	std::cout << result.ratio << std::endl;

	NN.set_classify_context(model, output);

	std::thread t1(&gui_thread, std::ref(i));

	while (i.is_active()) {
		output = i.wait_for_output();

		final.normalize_from(output);

		std::cout << (int) NN.classify(final) << std::endl;
		final.store("outputDraw");
	}

	t1.join();

	return 0;

}
