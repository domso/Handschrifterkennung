#ifndef data_sample_set_h
#define data_sample_set_h

#include <fstream>
#include <string>
#include "sample.h"
#include "../util/conv_endian.h"

namespace data {
namespace sample_set {
/**
 * Loads a dataset in the IDX-format into a vector of samples
 * See http://yann.lecun.com/exdb/mnist/
 * @param imgFileName: idx-file containing the sample-data
 * @param labelFileName: idx-file containing the labels for the samples
 * @return: std:.vector containing all successful loaded samples (empty on error)
 */
template<typename imgT>
std::vector<sample<imgT>> load(const std::string& imgFileName,
		const std::string& labelFileName) {
	std::ifstream imgFile;
	std::ifstream labelFile;
	std::vector<sample<imgT>> samples;
	int32_t magicImg;
	int32_t magicLabel;
	int32_t numItemImg;
	int32_t numItemLabel;
	int32_t width;
	int32_t height;

	imgFile.open(imgFileName);
	labelFile.open(labelFileName);

	if (!imgFile.is_open() || !labelFile.is_open()) {
		return samples;
	}

	imgFile.read((char*) &magicImg, sizeof(magicImg));
	imgFile.read((char*) &numItemImg, sizeof(numItemImg));
	imgFile.read((char*) &width, sizeof(width));
	imgFile.read((char*) &height, sizeof(height));

	labelFile.read((char*) &magicLabel, sizeof(magicLabel));
	labelFile.read((char*) &numItemLabel, sizeof(numItemLabel));

	if (!imgFile.good() || !labelFile.good() || numItemImg != numItemLabel) {
		return samples;
	}

	magicImg = util::conv_endian<int32_t>(magicImg);
	numItemImg = util::conv_endian<int32_t>(numItemImg);
	width = util::conv_endian<int32_t>(width);
	height = util::conv_endian<int32_t>(height);

	magicLabel = util::conv_endian<int32_t>(magicLabel);
	numItemLabel = util::conv_endian<int32_t>(numItemLabel);

	for (int i = 0; i < numItemImg; i++) {
		sample<imgT> s(width, height);

		for (int pi = 0; pi < width * height; pi++) {
			uint8_t pixel;

			if (!imgFile.read((char*) &pixel, sizeof(pixel))) {
				break;
			}

			s[pi] = (double) pixel / (double) 256;
		}

		uint8_t label;

		if (!labelFile.read((char*) &label, sizeof(label))) {
			break;
		}

		s.set_label(label);
		samples.push_back(std::move(s));
	}

	return samples;
}
}
}

#endif

