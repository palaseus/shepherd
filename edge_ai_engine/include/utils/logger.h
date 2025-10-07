/**
 * @file logger.h
 * @brief Logging utility interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the Logger class for logging messages in the Edge AI Engine.
 */

#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <memory>

namespace EdgeAI {

/**
 * @enum LogLevel
 * @brief Log levels
 */
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

/**
 * @class Logger
 * @brief Singleton logger class
 * 
 * The Logger class provides a thread-safe logging facility for the Edge AI Engine.
 */
class Logger {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the logger instance
     */
    static Logger& GetInstance();
    
    /**
     * @brief Set the log level
     * @param level Log level
     */
    void SetLogLevel(LogLevel level);
    
    /**
     * @brief Get the current log level
     * @return Current log level
     */
    LogLevel GetLogLevel() const;
    
    /**
     * @brief Set log file
     * @param file_path Path to log file
     */
    void SetLogFile(const std::string& file_path);
    
    /**
     * @brief Enable or disable console logging
     * @param enable Enable console logging
     */
    void SetLogToConsole(bool enable);
    
    /**
     * @brief Enable or disable file logging
     * @param enable Enable file logging
     */
    void SetLogToFile(bool enable);
    
    /**
     * @brief Log a message
     * @param level Log level
     * @param message Message to log
     */
    void Log(LogLevel level, const std::string& message);
    
    /**
     * @brief Log a debug message
     * @param message Message to log
     */
    void Debug(const std::string& message);
    
    /**
     * @brief Log an info message
     * @param message Message to log
     */
    void Info(const std::string& message);
    
    /**
     * @brief Log a warning message
     * @param message Message to log
     */
    void Warning(const std::string& message);
    
    /**
     * @brief Log an error message
     * @param message Message to log
     */
    void Error(const std::string& message);
    
    /**
     * @brief Log a fatal message
     * @param message Message to log
     */
    void Fatal(const std::string& message);

    // Constructor and destructor need to be public for make_unique
    Logger();
    ~Logger();

private:
    
    // Disable copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    LogLevel log_level_;
    bool log_to_file_;
    bool log_to_console_;
    std::ofstream log_file_;
    mutable std::mutex mutex_;
    
    static std::unique_ptr<Logger> instance_;
    static std::mutex instance_mutex_;
    
    /**
     * @brief Convert log level to string
     * @param level Log level
     * @return String representation of log level
     */
    std::string LogLevelToString(LogLevel level);
};

// Convenience macros
#define EDGE_AI_LOG_DEBUG(message) EdgeAI::Logger::GetInstance().Debug(message)
#define EDGE_AI_LOG_INFO(message) EdgeAI::Logger::GetInstance().Info(message)
#define EDGE_AI_LOG_WARNING(message) EdgeAI::Logger::GetInstance().Warning(message)
#define EDGE_AI_LOG_ERROR(message) EdgeAI::Logger::GetInstance().Error(message)
#define EDGE_AI_LOG_FATAL(message) EdgeAI::Logger::GetInstance().Fatal(message)

} // namespace EdgeAI
