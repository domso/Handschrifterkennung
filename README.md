Handwritten digit recognition with CUDA and C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses the "THE MNIST DATABASE of handwritten digits" to learn an artificial neural network
using "Feed-Forward" and "Back-Propagation" for recognition of handwritten digits.

Usage:
./Handschrifterkennung [CONFIG]

[CONFIG]
	Path to the config-file

Configuration: [key][:|=][value]

General:
	[implementation]
		Currently [c++] and [cuda] is supported

	[path_training_samples]
		Path to the training-samples

	[path_training_labels]
		Path to the training-labels

	[path_testing_samples]
		Path to the test-samples

	[path_testing_labels]
		Path to the test-labels

	[num_threads]
		Number of threads for calculation
 		This field will be ignored in the cuda-implementation
		Default: 1

Learning:
	[num_hidden]
		Number of nodes in the hidden-layer
		Default: 20

	[num_relearning]
		Number of learning-iterations on the same trainings-set
		Default: 1

	[learning_rate]
		Learning-Rate for the back-propagation algorithm
		Default: 1

Interface:
	[use_gui]
		Activate GUI-window for a live-demo
		Default: 1

	[window_width]
		X-Resolution of the GUI-window
		Default: 800

	[window_height]
		Y-Resolution of the GUI-window
		Default: 800

