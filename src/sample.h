#ifndef data_sample_h
#define data_sample_h

#include <fstream>
#include <vector>
#include <stdint.h>
#include <string>
#include <iostream>
#include <random>

namespace data {
    template <typename imgT>
    class sample {
    public:

        sample(const int id, const int width, const int height) : id_(id), width_(width), height_(height), internalData_(width_ * height_) {

        }

        imgT& operator [](const int index) {
            return internalData_[index];
        }

        const imgT& operator [](const int index) const {
			return internalData_[index];
		}

        bool operator ==(const sample<imgT>& o) const {
        	return internalData_ == o.internalData_;
        }

        bool operator !=(const sample<imgT>& o) const {
           	return internalData_ != o.internalData_;
        }

        int get_width() const {
            return width_;
        }

        int get_height() const {
            return height_;
        }

        uint8_t get_label() const {
            return label_;
        }

        void set_label(const uint8_t newLabel) {
            label_ = newLabel;
        }

        void print() const {
            std::cout << "sample-" << id_ << ": " << "(" << width_ << " x " << height_ << ") | " << (int) label_ << std::endl;
        }

        bool store(const std::string& filenamePrefix) {
            std::ofstream file;
            file.open(filenamePrefix + "_#" + std::to_string(id_) + "_" + std::to_string((int)label_) + ".pgm");

            if (file.is_open()) {
                //https://de.wikipedia.org/wiki/Portable_Anymap
                file << "P5 ";
                file << std::to_string(width_);
                file << " ";
                file << std::to_string(height_);
                file << " 255 ";

                for (imgT p : internalData_) {
                    file << (uint8_t)(p * 255);
                }

                return file.good();
            }

            return false;
        }

        int size() const {
        	return internalData_.size();
        }

        std::vector<imgT>& internalData() {
        	return internalData_;
        }

        const std::vector<imgT>& internalData() const{
           	return internalData_;
        }

        void normalize_from(const sample<imgT>& original) {
        	sample<imgT> tmp(0, original.get_width(), original.get_height());
        	float max = normalize_smooth_and_max(tmp, original);
        	normalize_rescale(tmp, original, max);
        	normalize_center(*this, tmp);
        }

    private:

        float normalize_smooth_and_max(sample<float>& output, const sample<float>& input) {
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

					value = value * (1 - (std::rand() % 100000) / (double)1000000);
					if (value > max) {
						max = value;
					}

					output[(y + 0) * 28 + (x + 0)] = value;
				}
			}

			return max;
        }

        void normalize_rescale(sample<float>& output, const sample<float>& input, float max) {
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

        void normalize_center(sample<float>& output, const sample<float>& input) {
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
					output[(y + (13 - offsetY)) * 28 + (x + (13 - offsetX))] = input[(y + 0) * 28 + (x + 0)];
				}
			}
        }



        int id_;
        int width_;
        int height_;
        uint8_t label_;
        std::vector<imgT> internalData_;
    };
}

#endif

