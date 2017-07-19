#ifndef data_sample_h
#define data_sample_h

#include <fstream>
#include <vector>
#include <stdint.h>
#include <string>
#include <iostream>

namespace data {
/*
 * Basic storage-class for a single sample
 */
template<typename imgT>
class sample {
public:
	/*
	 * Initializes a new sample with the given arguments
	 * @param width: x-resolution of the sample-image
	 * @param height: y-resolution of the sample-image
	 */
	sample(const int width, const int height) :
			m_width(width), m_height(height), m_internalData(width * height) {
	}

	/*
	 * Returns the pixel-data at a given index
	 * @param index: row-major index of requested pixel
	 * @return: reference to the requested pixel
	 */
	imgT& operator [](const int index) {
		return m_internalData[index];
	}

	/*
	 * Returns the pixel-data at a given index
	 * @param index: row-major index of requested pixel
	 * @return: reference to the requested pixel
	 */
	const imgT& operator [](const int index) const {
		return m_internalData[index];
	}

	/*
	 * Compares the own pixel-data with the given sample
	 * @param o: sample to compare with
	 * @return: true if both samples are equal
	 */
	bool operator ==(const sample<imgT>& o) const {
		return m_internalData == o.m_internalData;
	}

	/*
	 * Compares the own pixel-data with the given sample
	 * @param o: sample to compare with
	 * @return: true if both samples are not equal
	 */
	bool operator !=(const sample<imgT>& o) const {
		return m_internalData != o.m_internalData;
	}

	/*
	 * @return: x-resolution of the sample-image
	 */
	int get_width() const {
		return m_width;
	}

	/*
	 * @return: y-resolution of the sample-image
	 */
	int get_height() const {
		return m_height;
	}

	/*
	 * @return: the label of the sample
	 */
	uint8_t get_label() const {
		return m_label;
	}

	/*
	 * Sets a new label for the sample
	 * @param newLabel: the label
	 */
	void set_label(const uint8_t newLabel) {
		m_label = newLabel;
	}

	/*
	 * Prints the resolution and label to std::cout
	 */
	void print() const {
		std::cout << "sample" << ": " << "(" << m_width << " x " << m_height
				<< ") | " << (int) m_label << std::endl;
	}

	/*
	 * Exports the sample to an Portable_Anymap(.pgm)-file
	 * --> https://de.wikipedia.org/wiki/Portable_Anymap
	 * @param filenamePrefix: the prefix for the output-file($(prefix)_label.pgm)
	 * @return: true on success
	 */
	bool store(const std::string& filenamePrefix) {
		std::ofstream file;
		file.open(
				filenamePrefix + "_" + std::to_string((int) m_label) + ".pgm");

		if (file.is_open()) {
			file << "P5 ";
			file << std::to_string(m_width);
			file << " ";
			file << std::to_string(m_height);
			file << " 255 ";

			for (imgT p : m_internalData) {
				file << (uint8_t) (p * 255);
			}

			return file.good();
		}

		return false;
	}

	/*
	 * @return: the number of pixels (width * height)
	 */
	int size() const {
		return m_internalData.size();
	}

	/*
	 * @return: reference to the internal data representation
	 */
	std::vector<imgT>& internal_data() {
		return m_internalData;
	}

	/*
	 * @return: reference to the internal data representation
	 */
	const std::vector<imgT>& internal_data() const {
		return m_internalData;
	}

	/*
	 * The MNIST dataset comes with a huge bias on color, thickness
	 * and other parameters. In order to allow "self-made" samples
	 * to have a decent performance, a normalized sample needs to be
	 * created by this function.
	 * The Normalization includes:
	 * 	- blur by a modified 3x3-Gaussian-Filter (see normalize_smooth_and_max)
	 * 	- Randomize with a small Noise-Signal (see normalize_smooth_and_max)
	 * 	- Rescale to [0, 1] (see normalize_rescale)
	 * 	- Upscale (see normalize_rescale)
	 * 	- Reusage of the original pixels with value = 1 (normalize_rescale)
	 * 	- Center the non-zero pixels in the center (normalize_center)
	 *
	 * @param original: original image stored as a sample (required pixel-range [0, 1])
	 */
	void normalize_from(const sample<imgT>& original) {
		sample<imgT> tmp(original.get_width(), original.get_height());
		float max = normalize_smooth_and_max(tmp, original);
		normalize_rescale(tmp, original, max);
		normalize_center(*this, tmp);
	}

private:

	/*
	 * See normalize_from()
	 * Performs a Gaussian-Blur on input, adds a small Noise-Signal
	 * and stores the result in output
	 * Modified Gaussian-Kernel: (dont ask, it works better)
	 * 121
	 * 282
	 * 121
	 *
	 * @param input: original image stored as a sample
	 * @param output: new image stored as a sample
	 * @return: the maximal pixel-data in range [0, 1]
	 */
	static float normalize_smooth_and_max(sample<float>& output,
			const sample<float>& input) {
		std::srand(0);
		float value;
		float max = 0;
		for (int y = 1; y < 27; y++) {
			for (int x = 1; x < 27; x++) {
				value = 0;
				value += input[(y - 1) * 28 + (x - 1)] * 1;
				value += input[(y - 1) * 28 + (x + 0)] * 2;
				value += input[(y - 1) * 28 + (x + 1)] * 1;
				value += input[(y + 0) * 28 + (x - 1)] * 2;
				value += input[(y + 0) * 28 + (x + 0)] * 8;
				value += input[(y + 0) * 28 + (x + 1)] * 2;
				value += input[(y + 1) * 28 + (x - 1)] * 1;
				value += input[(y + 1) * 28 + (x + 0)] * 2;
				value += input[(y + 1) * 28 + (x + 1)] * 1;
				value /= 20;

				value = value * (1 - (std::rand() % 100000) / (double) 1000000);
				if (value > max) {
					max = value;
				}

				output[(y + 0) * 28 + (x + 0)] = value;
			}
		}

		return max;
	}

	/*
	 * See normalize_from()
	 * Performs a Rescale to [0, 1] on input, upscales all value
	 * and copies all pixel from the input with a value = 1.
	 * The result is stored in output
	 * @param input: original image stored as a sample
	 * @param output: new image stored as a sample
	 */
	static void normalize_rescale(sample<float>& output,
			const sample<float>& input, float max) {
		for (int y = 1; y < 27; y++) {
			for (int x = 1; x < 27; x++) {
				output[(y + 0) * 28 + (x + 0)] *= (1.0 / max);
				output[(y + 0) * 28 + (x + 0)] *= 3.5;

				if (output[(y + 0) * 28 + (x + 0)] > 1) {
					output[(y + 0) * 28 + (x + 0)] = 1;
				}

				if (input[(y + 0) * 28 + (x + 0)] == 1) {
					output[(y + 0) * 28 + (x + 0)] = 1;
				}
			}
		}
	}

	/*
	 * See normalize_from()
	 * Moves the non-zero pixels from input to the center
	 * and stores the result in output
	 * @param input: original image stored as a sample
	 * @param output: new image stored as a sample
	 */
	static void normalize_center(sample<float>& output,
			const sample<float>& input) {
		int minX = 28;
		int maxX = 0;
		int minY = 28;
		int maxY = 0;

		for (int y = 1; y < 27; y++) {
			for (int x = 1; x < 27; x++) {
				if (input[(y + 0) * 28 + (x + 0)] != 0) {
					if (x < minX) {
						minX = x;
					}
					if (x > maxX) {
						maxX = x;
					}
					if (y < minY) {
						minY = y;
					}
					if (y > maxY) {
						maxY = y;
					}
				}
				output[(y + 0) * 28 + (x + 0)] = 0;
			}
		}

		int offsetX = minX + (maxX - minX) / 2;
		int offsetY = minY + (maxY - minY) / 2;

		for (int y = minY; y <= maxY; y++) {
			for (int x = minX; x <= maxX; x++) {
				output[(y + (13 - offsetY)) * 28 + (x + (13 - offsetX))] =
						input[(y + 0) * 28 + (x + 0)];
			}
		}
	}

	int m_width;
	int m_height;
	uint8_t m_label;
	std::vector<imgT> m_internalData;
};
}

#endif

