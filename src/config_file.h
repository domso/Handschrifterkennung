#ifndef general_util_config_file_h
#define general_util_config_file_h

#include <string>
#include <unordered_map>

namespace util {
    //______________________________________________________________________________________________________
    //
    // simple class to load, parse, validate and access a configuration file
    //______________________________________________________________________________________________________
    class config_file {
    public:
        config_file();
        ~config_file();
        //______________________________________________________________________________________________________
        //
        // Description:
        // - tries to open the file specified by filename
        // - every line will be parsed into a key-value pair with ':' as seperator
        // - empty space will be ignored
        // Parameter:
        // - filename: the absolut or relative path to the config file
        //______________________________________________________________________________________________________
        void load(std::string filename);
        //______________________________________________________________________________________________________
        //
        // Description:
        // - if the loaded config file does not contain a specific string-entry, an internal error will be generated (see isError() and getErrorMsgs())
        // - exspects the value of the entry specified by key as a string
        // Parameter:
        // - key: the required key
        //______________________________________________________________________________________________________
        void requireString(const std::string& key);
        //______________________________________________________________________________________________________
        //
        // Description:
        // - if the loaded config file does not contain a specific string-entry, an internal warning will be generated (see isWarning() and getWarningMsgs())
        // - any missing recommend entry will be added with defaultValue as value
        // - exspects the value of the entry specified by key as a string
        // Parameters:
        // - key: the recommend key
        // - defaultValue: the default value for missing entries
        //______________________________________________________________________________________________________
        void recommendString(const std::string& key, const std::string& defaultValue = "");
        //______________________________________________________________________________________________________
        //
        // Description:
        // - Returns the string-value of the entry specified by the given key
        // - if the entry is not available, defaultValue will be returned, but the internal value remains unchanged
        // - exspects the value of the entry specified by key as a string
        // Parameters:
        // - key: the searched-key
        // - defaultValue: the default value for missing entries
        // Return:
        // - the entry-value  | for an existing entry
        // - the defaultValue | for a missing entry
        //______________________________________________________________________________________________________
        std::string getString(const std::string& key, const std::string& defaultValue = "") const;
        //______________________________________________________________________________________________________
        //
        // Description:
        // - if any requireString() or requireNumeric() call fails, the function will return true
        // Return:
        // - true  | on any error
        // - false | no error
        //______________________________________________________________________________________________________
        bool isError() const;
        //______________________________________________________________________________________________________
        //
        // Description:
        // - if any requireString() or requireNumeric() call fails, the function will return a string containing ALL generated error messages!
        // - the messages itself wont be deleted
        // Returns:
        // - string containing all available error messages
        //______________________________________________________________________________________________________
        const std::string& getErrorMsgs() const;
        //______________________________________________________________________________________________________
        //
        // Description:
        // - if any recommendString() or recommendNumeric() call fails, the function will return true
        // Return:
        // - true  | on any warning
        // - false | no warning
        //______________________________________________________________________________________________________
        bool isWarning() const;
        //______________________________________________________________________________________________________
        //
        // Description:
        // - if any recommendString() or recommendNumeric() call fails, the function will return a string containing ALL generated warning messages!
        // - the messages itself wont be deleted
        // Returns:
        // - string containing all available warning messages
        //______________________________________________________________________________________________________
        const std::string& getWarningMsgs() const;
        //______________________________________________________________________________________________________
        //
        // Description:
        // - clears all errors and warnings
        //______________________________________________________________________________________________________
        void clear();
        //______________________________________________________________________________________________________
        //
        // Description:
        // - if the loaded config file does not contain a specific numeric-entry, an internal error will be generated (see isError() and getErrorMsgs())
        // - exspects the value of the entry specified by key as an instance of the numeric type T
        // Parameter:
        // - key: the required key
        //______________________________________________________________________________________________________
        template <typename T>
        void requireNumeric(const std::string& key) {
            if (filename_.empty()) {
                return;
            }

            auto result = map_.find(key);

            if (result != map_.end()) {
                if (result->second != "") {
                    char* endPtr = nullptr;
                    double value = strtod(result->second.c_str(), &endPtr);

                    if (endPtr != result->second.c_str()) {
                        if ((T)value != value) {
                            result->second = std::to_string((T)value);
                            warningMsg_ += "[WARNING] Wrong numeric format for '" + key + "'";
                            warningMsg_ += " in " + filename_ + ".";
                            warningMsg_ += " Will use '" + result->second + "' instead.\n";
                        }

                        return;
                    }
                }

                errorMsg_ += "[ERROR] Require '" + key + "' as numeric value.\n";
                return;
            }

            errorMsg_ += "[ERROR] Require '" + key + "'";
            errorMsg_ += " in " + filename_ + ".\n";
        }
        //______________________________________________________________________________________________________
        //
        // Description:
        // - if the loaded config file does not contain a specific numeric-entry, an internal warning will be generated (see isWarning() and getWarningMsgs())
        // - any missing recommend entry will be added with defaultValue as value
        // - exspects the value of the entry specified by key as an instance of the numeric type T
        // - if the existing value is a general numeric value, but not of numeric type T, the internal entry will be changed to a casted version of the numeric entry and a warning is generated
        // Parameters:
        // - key: the recommend key
        // - defaultValue: the default value for missing entries
        //______________________________________________________________________________________________________
        template <typename T>
        void recommendNumeric(const std::string& key, const T defaultValue = -1) {
            if (filename_.empty()) {
                return;
            }

            auto result = map_.find(key);

            if (result != map_.end()) {
                if (result->second != "") {
                    char* endPtr = nullptr;
                    double value = strtod(result->second.c_str(), &endPtr);

                    if (endPtr != result->second.c_str()) {
                        // simple 'hack' to verify that value is a instance of T
                        if ((T)value != value) {
                            result->second = std::to_string((T)value);
                            warningMsg_ += "[WARNING] Wrong numeric format for '" + key + "'";
                            warningMsg_ += " in " + filename_ + ".";
                            warningMsg_ += " Will use '" + result->second + "' instead.\n";
                        }

                        return;
                    }
                }

                warningMsg_ += "[WARNING] Ignore non-numeric value '" + result->second + "'";
                warningMsg_ += " for '" + key + "'.\n";
            }

            map_[key] = std::to_string(defaultValue);
            warningMsg_ += "[WARNING] Recommend '" + key + "'";
            warningMsg_ += " (Default = " + std::to_string(defaultValue) + ")";
            warningMsg_ += " in " + filename_ + ".\n";
        }
        //______________________________________________________________________________________________________
        //
        // Description:
        // - Returns the numeric-value of the entry specified by the given key
        // - if the entry is not available, defaultValue will be returned, but the internal value remains unchanged
        // - exspects the value of the entry specified by key as an instance of the numeric type T
        // Parameters:
        // - key: the searched-key
        // - defaultValue: the default value for missing entries
        // Return:
        // - the entry-value  | for an existing entry
        // - the defaultValue | for a missing entry
        //______________________________________________________________________________________________________
        template <typename T>
        T getNumeric(const std::string& key, const T defaultValue = -1) const {
            if (filename_.empty()) {
                return defaultValue;
            }

            auto result = map_.find(key);

            if (result != map_.end()) {
                if (result->second != "") {
                    char* endPtr = nullptr;
                    double value = strtod(result->second.c_str(), &endPtr);

                    if (endPtr != result->second.c_str()) {
                        if ((T)value == value) {
                            return (T)value;
                        }
                    }
                }
            }

            return defaultValue;
        }
    private:
        //______________________________________________________________________________________________________
        //
        // Description:
        // - tries to extract new entry from the given string
        // - the string will be parsed into a key-value pair with ':' as seperator
        // - empty space will be ignored
        // Parameter:
        // - input: string containing a line from a config file
        // Return:
        // - true  | on success
        // - false | on any error
        //______________________________________________________________________________________________________
        bool insert(const std::string input);

        // the filename of last loaded config file
        std::string filename_;
        // string containing all error messages
        std::string errorMsg_;
        // string containing all warning messages
        std::string warningMsg_;
        // internal representation of the key-value pair
        std::unordered_map<std::string, std::string> map_;
    };
}

#endif

