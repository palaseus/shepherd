/**
 * @file config_parser.h
 * @brief Configuration parser utility interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the ConfigParser class for parsing configuration files in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace EdgeAI {

/**
 * @class ConfigParser
 * @brief Configuration parser utility class
 * 
 * The ConfigParser class provides functionality for parsing and managing configuration files.
 */
class ConfigParser {
public:
    /**
     * @brief Constructor
     */
    ConfigParser();
    
    /**
     * @brief Destructor
     */
    ~ConfigParser();
    
    // Disable copy constructor and assignment operator
    ConfigParser(const ConfigParser&) = delete;
    ConfigParser& operator=(const ConfigParser&) = delete;
    
    /**
     * @brief Load configuration from file
     * @param file_path Path to configuration file
     * @return Status indicating success or failure
     */
    edge_ai::Status LoadFromFile(const std::string& file_path);
    
    /**
     * @brief Save configuration to file
     * @param file_path Path to configuration file
     * @return Status indicating success or failure
     */
    edge_ai::Status SaveToFile(const std::string& file_path) const;
    
    /**
     * @brief Get string value
     * @param key Configuration key
     * @param default_value Default value if key not found
     * @return String value
     */
    std::string GetString(const std::string& key, const std::string& default_value = "") const;
    
    /**
     * @brief Get integer value
     * @param key Configuration key
     * @param default_value Default value if key not found
     * @return Integer value
     */
    int GetInt(const std::string& key, int default_value = 0) const;
    
    /**
     * @brief Get double value
     * @param key Configuration key
     * @param default_value Default value if key not found
     * @return Double value
     */
    double GetDouble(const std::string& key, double default_value = 0.0) const;
    
    /**
     * @brief Get boolean value
     * @param key Configuration key
     * @param default_value Default value if key not found
     * @return Boolean value
     */
    bool GetBool(const std::string& key, bool default_value = false) const;
    
    /**
     * @brief Set string value
     * @param key Configuration key
     * @param value String value
     */
    void SetString(const std::string& key, const std::string& value);
    
    /**
     * @brief Set integer value
     * @param key Configuration key
     * @param value Integer value
     */
    void SetInt(const std::string& key, int value);
    
    /**
     * @brief Set double value
     * @param key Configuration key
     * @param value Double value
     */
    void SetDouble(const std::string& key, double value);
    
    /**
     * @brief Set boolean value
     * @param key Configuration key
     * @param value Boolean value
     */
    void SetBool(const std::string& key, bool value);
    
    /**
     * @brief Check if key exists
     * @param key Configuration key
     * @return True if key exists
     */
    bool HasKey(const std::string& key) const;
    
    /**
     * @brief Remove key
     * @param key Configuration key to remove
     */
    void RemoveKey(const std::string& key);
    
    /**
     * @brief Clear all configuration
     */
    void Clear();
    
    /**
     * @brief Get all keys
     * @return Vector of all configuration keys
     */
    std::vector<std::string> GetKeys() const;

private:
    std::unordered_map<std::string, std::string> config_;
};

} // namespace EdgeAI
