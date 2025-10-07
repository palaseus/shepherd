/**
 * @file config_parser.cpp
 * @brief Configuration parser utility implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "utils/config_parser.h"
#include "utils/logger.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace EdgeAI {

ConfigParser::ConfigParser() = default;

ConfigParser::~ConfigParser() = default;

edge_ai::Status ConfigParser::LoadFromFile(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            Logger::GetInstance().Error("Failed to open config file: " + file_path);
            return edge_ai::Status::FAILURE;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            // Parse key-value pairs
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                config_[key] = value;
            }
        }
        
        file.close();
        return edge_ai::Status::SUCCESS;
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error loading config file: " + std::string(e.what()));
        return edge_ai::Status::FAILURE;
    }
}

edge_ai::Status ConfigParser::SaveToFile(const std::string& file_path) const {
    try {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            Logger::GetInstance().Error("Failed to create config file: " + file_path);
            return edge_ai::Status::FAILURE;
        }
        
        for (const auto& pair : config_) {
            file << pair.first << "=" << pair.second << std::endl;
        }
        
        file.close();
        return edge_ai::Status::SUCCESS;
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error saving config file: " + std::string(e.what()));
        return edge_ai::Status::FAILURE;
    }
}

std::string ConfigParser::GetString(const std::string& key, const std::string& default_value) const {
    auto it = config_.find(key);
    if (it != config_.end()) {
        return it->second;
    }
    return default_value;
}

int ConfigParser::GetInt(const std::string& key, int default_value) const {
    auto it = config_.find(key);
    if (it != config_.end()) {
        try {
            return std::stoi(it->second);
        } catch (const std::exception& e) {
            Logger::GetInstance().Warning("Failed to parse int value for key " + key + ": " + it->second);
        }
    }
    return default_value;
}

double ConfigParser::GetDouble(const std::string& key, double default_value) const {
    auto it = config_.find(key);
    if (it != config_.end()) {
        try {
            return std::stod(it->second);
        } catch (const std::exception& e) {
            Logger::GetInstance().Warning("Failed to parse double value for key " + key + ": " + it->second);
        }
    }
    return default_value;
}

bool ConfigParser::GetBool(const std::string& key, bool default_value) const {
    auto it = config_.find(key);
    if (it != config_.end()) {
        std::string value = it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        
        if (value == "true" || value == "1" || value == "yes") {
            return true;
        } else if (value == "false" || value == "0" || value == "no") {
            return false;
        } else {
            Logger::GetInstance().Warning("Failed to parse bool value for key " + key + ": " + it->second);
        }
    }
    return default_value;
}

void ConfigParser::SetString(const std::string& key, const std::string& value) {
    config_[key] = value;
}

void ConfigParser::SetInt(const std::string& key, int value) {
    config_[key] = std::to_string(value);
}

void ConfigParser::SetDouble(const std::string& key, double value) {
    config_[key] = std::to_string(value);
}

void ConfigParser::SetBool(const std::string& key, bool value) {
    config_[key] = value ? "true" : "false";
}

bool ConfigParser::HasKey(const std::string& key) const {
    return config_.find(key) != config_.end();
}

void ConfigParser::RemoveKey(const std::string& key) {
    config_.erase(key);
}

void ConfigParser::Clear() {
    config_.clear();
}

std::vector<std::string> ConfigParser::GetKeys() const {
    std::vector<std::string> keys;
    for (const auto& pair : config_) {
        keys.push_back(pair.first);
    }
    return keys;
}

} // namespace EdgeAI
