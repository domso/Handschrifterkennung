#ifndef parameters_h
#define parameters_h

namespace parameters {

struct implementation {
	constexpr static const auto key = "implementation";
	constexpr static const auto defaultValue = "";
};

struct path_training_samples {
	constexpr static const auto key = "path_training_samples";
	constexpr static const auto defaultValue = "";
};

struct path_training_labels {
	constexpr static const auto key = "path_training_labels";
	constexpr static const auto defaultValue = "";
};

struct path_testing_samples {
	constexpr static const auto key = "path_testing_samples";
	constexpr static const auto defaultValue = "";
};

struct path_testing_labels {
	constexpr static const auto key = "path_testing_labels";
	constexpr static const auto defaultValue = "";
};

struct use_gui {
	constexpr static const auto key = "use_gui";
	constexpr static const int defaultValue = 1;
};

struct num_hidden {
	constexpr static const auto key = "num_hidden";
	constexpr static const int defaultValue = 20;
};

struct num_relearning {
	constexpr static const auto key = "num_relearning";
	constexpr static const int defaultValue = 1;
};

struct learning_rate {
	constexpr static const auto key = "learning_rate";
	constexpr static const float defaultValue = 0.2;
};

struct window_width {
	constexpr static const auto key = "window_width";
	constexpr static const int defaultValue = 800;
};

struct window_height {
	constexpr static const auto key = "window_height";
	constexpr static const int defaultValue = 800;
};

struct num_threads {
	constexpr static const auto key = "num_threads";
	constexpr static const float defaultValue = 1;
};


}

#endif
