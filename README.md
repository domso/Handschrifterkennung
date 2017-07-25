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
	[implementation][int]
		Currently [c++] and [cuda] is supported

	[path_training_samples][string]
		Path to the training-samples

	[path_training_labels][string]
		Path to the training-labels

	[path_testing_samples][string]
		Path to the test-samples

	[path_testing_labels][string]
		Path to the test-labels

	[num_threads][int]
		Number of threads for calculation
 		This field will be ignored in the cuda-implementation
		Default: 1

Learning:
	[num_hidden][int]
		Number of nodes in the hidden-layer
		Default: 20

	[num_relearning][float]
		Number of learning-iterations on the same trainings-set
		Default: 1

	[learning_rate][float]
		Learning-Rate for the back-propagation algorithm
		Default: 1

Interface:
	[use_gui][int]
		Activate GUI-window for a live-demo
		Default: 1

	[window_width][int]
		X-Resolution of the GUI-window
		Default: 800

	[window_height][int]
		Y-Resolution of the GUI-window
		Default: 800

	[use_accelerator][int]
		Enable hardware accelerator for GUI
		Default: 1		

