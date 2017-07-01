#ifndef data_sample_h
#define data_sample_h

#include <fstream>
#include <vector>
#include <stdint.h>
#include <string>

namespace data {
    template <typename imgT>
    class sample {
    public:

        sample(const int id, const int width, const int height) : id_(id), width_(width), height_(height), internalData_(width_ * height_) {

        }

        imgT& operator [](const int index) {
            return internalData_[index];
        }

        int getWidth() const {
            return width_;
        }

        int getHeight() const {
            return height_;
        }

        uint8_t getLabel() const {
            return label_;
        }

        void setLabel(const uint8_t newLabel) {
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
                    file << p;
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

    private:
        int id_;
        int width_;
        int height_;
        uint8_t label_;
        std::vector<imgT> internalData_;
    };
}

#endif

