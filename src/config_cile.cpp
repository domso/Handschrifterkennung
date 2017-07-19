#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include "config_file.h"

namespace util {

    config_file::config_file() {

    }

    config_file::~config_file() {

    }

    void config_file::load(std::string filename) {
        std::ifstream file;
        std::string line;

        file.open(filename);

        if (file.is_open()) {
            while (std::getline(file, line)) {
                insert(line);
            }

            file.close();

            filename_ = filename;
            return;
        }

        filename_ = "";
        errorMsg_ += "[ERROR] Could not open '" + filename + "' as config file.";
        return;
    }

    void config_file::requireString(const std::string& key) {
        if (filename_.empty()) {
            return;
        }

        auto result = map_.find(key);

        if (result != map_.end()) {
            if (result->second != "") {
                return;
            }
        }

        errorMsg_ += "[ERROR] Require '" + key + "'";
        errorMsg_ += " in " + filename_ + ".\n";
    }

    void config_file::recommendString(const std::string& key, const std::string& defaultValue) {
        if (filename_.empty()) {
            return;
        }

        auto result = map_.find(key);

        if (result != map_.end()) {
            if (result->second != "") {
                return;
            }
        }

        map_[key] = defaultValue;
        warningMsg_ += "[WARNING] Recommend '" + key + "'";
        warningMsg_ += " (Default = '" + defaultValue + "')";
        warningMsg_ += " in " + filename_ + ".\n";
    }

    std::string config_file::getString(const std::__cxx11::string& key, const std::__cxx11::string& defaultValue) const {
        auto result = map_.find(key);

        if (result != map_.end()) {
            if (result->second != "") {
                return result->second;
            }
        }

        return defaultValue;
    }

    bool config_file::isError() const {
        return !errorMsg_.empty();
    }

    const std::string& config_file::getErrorMsgs() const {
        return errorMsg_;
    }

    bool config_file::isWarning() const {
        return !warningMsg_.empty();
    }

    const std::string& config_file::getWarningMsgs() const {
        return warningMsg_;
    }

    void config_file::clear() {
        errorMsg_ = "";
        warningMsg_ = "";
    }

    bool config_file::insert(const std::string input) {
        int mode = 0;
        int num_chars[2];
        num_chars[0] = 0;
        num_chars[1] = 0;

        for (int i = 0; i < input.length(); i++) {
            if (input[i] == ':' || input[i] == '=') {
                mode = 1;
            } else if (input[i] != ' ') {
                num_chars[mode]++;
            }
        }

        if (mode == 0 || num_chars[0] == 0 || num_chars[1] == 0) {
            return false;
        }

        std::string components[2];

        components[0].resize(num_chars[0], ' ');
        components[1].resize(num_chars[1], ' ');

        mode = 0;
        num_chars[0] = 0;
        num_chars[1] = 0;

        for (int i = 0; i < input.length(); i++) {
            if (input[i] == ':'|| input[i] == '=') {
                mode = 1;
            } else if (input[i] != ' ') {
                components[mode][num_chars[mode]] = input[i];
                num_chars[mode]++;
            }
        }

        if (components[0][0] != '#') {
            map_.insert({components[0], components[1]});
        }

        return true;
    }

}

