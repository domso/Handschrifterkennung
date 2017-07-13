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

        sample(const int id, const int width, const int height) : m_id(id), m_width(width), m_height(height), m_internalData(width * height) {

        }

        imgT& operator [](const int index) {
            return m_internalData[index];
        }

        const imgT& operator [](const int index) const {
			return m_internalData[index];
		}

        bool operator ==(const sample<imgT>& o) const {
        	return m_internalData == o.m_internalData;
        }

        bool operator !=(const sample<imgT>& o) const {
           	return m_internalData != o.m_internalData;
        }

        int get_width() const {
            return m_width;
        }

        int get_height() const {
            return m_height;
        }

        uint8_t get_label() const {
            return m_label;
        }

        void set_label(const uint8_t newLabel) {
            m_label = newLabel;
        }

        void print() const {
            std::cout << "sample-" << m_id << ": " << "(" << m_width << " x " << m_height << ") | " << (int) m_label << std::endl;
        }

        bool store(const std::string& filenamePrefix) {
            std::ofstream file;
            file.open(filenamePrefix + "_#" + std::to_string(m_id) + "_" + std::to_string((int)m_label) + ".pgm");

            if (file.is_open()) {
                //https://de.wikipedia.org/wiki/Portable_Anymap
                file << "P5 ";
                file << std::to_string(m_width);
                file << " ";
                file << std::to_string(m_height);
                file << " 255 ";

                for (imgT p : m_internalData) {
                    file << (uint8_t)(p * 255);
                }

                return file.good();
            }

            return false;
        }

        int size() const {
        	return m_internalData.size();
        }

        std::vector<imgT>& internal_data() {
        	return m_internalData;
        }

        const std::vector<imgT>& internal_data() const{
           	return m_internalData;
        }

        void normalize_from(const sample<imgT>& original) {
        	sample<imgT> tmp(0, original.get_width(), original.get_height());
        	float max = normalize_smooth_and_max(tmp, original);
        	normalize_rescale(tmp, original, max);
        	normalize_center(*this, tmp);
        }

    private:

        static float normalize_smooth_and_max(sample<float>& output, const sample<float>& input) {
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

        static void normalize_rescale(sample<float>& output, const sample<float>& input, float max) {
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

        static void normalize_center(sample<float>& output, const sample<float>& input) {
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

        int m_id;
        int m_width;
        int m_height;
        uint8_t m_label;
        std::vector<imgT> m_internalData;
    };
}

#endif

