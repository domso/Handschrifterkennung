#ifndef parameters_h
#define parameters_h

namespace parameters {

/*
 * Currently 'c++' and 'cuda' is supported
 */
struct implementation {
	constexpr static const auto key = "implementation";
	constexpr static const auto defaultValue = "";
};

/*
 * Path to the training-samples
 */
struct path_training_samples {
	constexpr static const auto key = "path_training_samples";
	constexpr static const auto defaultValue = "";
};

/*
 * Path to the training-labels
 */
struct path_training_labels {
	constexpr static const auto key = "path_training_labels";
	constexpr static const auto defaultValue = "";
};

/*
 * Path to the test-samples
 */
struct path_testing_samples {
	constexpr static const auto key = "path_testing_samples";
	constexpr static const auto defaultValue = "";
};

/*
 * Path to the test-labels
 */
struct path_testing_labels {
	constexpr static const auto key = "path_testing_labels";
	constexpr static const auto defaultValue = "";
};

/*
 * Number of nodes in the hidden-layer
 */
struct num_hidden {
	constexpr static const auto key = "num_hidden";
	constexpr static const int defaultValue = 20;
};

/*
 * Number of threads for calculation
 * This field will be ignored in the cuda-implementation
 */
struct num_threads {
	constexpr static const auto key = "num_threads";
	constexpr static const float defaultValue = 1;
};

/*
 * Number of learning-iterations on the same trainings-set
 */
struct num_relearning {
	constexpr static const auto key = "num_relearning";
	constexpr static const int defaultValue = 1;
};

/*
 * Learning-Rate for the back-propagation algorithm
 */
struct learning_rate {
	constexpr static const auto key = "learning_rate";
	constexpr static const float defaultValue = 0.2;
};

/*
 * Activate GUI-window for a live-demo
 */
struct use_gui {
	constexpr static const auto key = "use_gui";
	constexpr static const int defaultValue = 1;
};

/*
 * X-Resolution of the GUI-window
 */
struct window_width {
	constexpr static const auto key = "window_width";
	constexpr static const int defaultValue = 800;
};

/*
 * Y-Resolution of the GUI-window
 */
struct window_height {
	constexpr static const auto key = "window_height";
	constexpr static const int defaultValue = 800;
};

}

#endif
