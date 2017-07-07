#ifndef data_sample_set_h
#define data_sample_set_h

#include <iostream>
#include <fstream>
#include <string>
#include "sample.h"
#include "helper.h"

namespace data {
    namespace sample_set {
        template <typename imgT>
        std::vector<sample<imgT>> load(const std::string& imgFileName, const std::string& labelFileName) {
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
            	std::cout << "could not open file!\n";
                return samples;
            }

            imgFile.read((char*)&magicImg, sizeof(magicImg));
            imgFile.read((char*)&numItemImg, sizeof(numItemImg));
            imgFile.read((char*)&width, sizeof(width));
            imgFile.read((char*)&height, sizeof(height));

            labelFile.read((char*)&magicLabel, sizeof(magicLabel));
            labelFile.read((char*)&numItemLabel, sizeof(numItemLabel));

            if (!imgFile.good() || !labelFile.good() || numItemImg != numItemLabel) {
            	std::cout << "bad file!\n";
                return samples;
            }

            magicImg = helper::convEndian<int32_t>(magicImg);
            numItemImg = helper::convEndian<int32_t>(numItemImg);
            width = helper::convEndian<int32_t>(width);
            height = helper::convEndian<int32_t>(height);

            magicLabel = helper::convEndian<int32_t>(magicLabel);
            numItemLabel = helper::convEndian<int32_t>(numItemLabel);

            for (int i = 0; i < numItemImg; i++) {
                sample<imgT> s(i, width, height);

                for (int pi = 0; pi < width * height; pi++) {
                    uint8_t pixel;

                    if (!imgFile.read((char*)&pixel, sizeof(pixel))) {
                        break;
                    }

                    s[pi] = pixel;
                }

                uint8_t label;

                if (!labelFile.read((char*)&label, sizeof(label))) {
                    break;
                }

                s.setLabel(label);
                samples.push_back(std::move(s));
            }

            return samples;
        }
    }
}

#endif
