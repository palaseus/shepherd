/**
 * @file logger.cpp
 * @brief Logging utility implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "utils/logger.h"
#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace EdgeAI {

// Global logger instance
std::unique_ptr<Logger> Logger::instance_ = nullptr;
std::mutex Logger::instance_mutex_;

Logger::Logger()
    : log_level_(LogLevel::INFO)
    , log_to_file_(false)
    , log_to_console_(true) {
}

Logger::~Logger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

Logger& Logger::GetInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::make_unique<Logger>();
    }
    return *instance_;
}

void Logger::SetLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    log_level_ = level;
}

LogLevel Logger::GetLogLevel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return log_level_;
}

void Logger::SetLogFile(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (log_file_.is_open()) {
        log_file_.close();
    }
    
    log_file_.open(file_path, std::ios::app);
    log_to_file_ = log_file_.is_open();
}

void Logger::SetLogToConsole(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    log_to_console_ = enable;
}

void Logger::SetLogToFile(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    log_to_file_ = enable;
}

void Logger::Log(LogLevel level, const std::string& message) {
    if (level < log_level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Format log message
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    ss << " [" << LogLevelToString(level) << "] " << message << std::endl;
    
    std::string formatted_message = ss.str();
    
    // Log to console
    if (log_to_console_) {
        if (level >= LogLevel::ERROR) {
            std::cerr << formatted_message;
        } else {
            std::cout << formatted_message;
        }
    }
    
    // Log to file
    if (log_to_file_ && log_file_.is_open()) {
        log_file_ << formatted_message;
        log_file_.flush();
    }
}

void Logger::Debug(const std::string& message) {
    Log(LogLevel::DEBUG, message);
}

void Logger::Info(const std::string& message) {
    Log(LogLevel::INFO, message);
}

void Logger::Warning(const std::string& message) {
    Log(LogLevel::WARNING, message);
}

void Logger::Error(const std::string& message) {
    Log(LogLevel::ERROR, message);
}

void Logger::Fatal(const std::string& message) {
    Log(LogLevel::FATAL, message);
}

std::string Logger::LogLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

} // namespace EdgeAI
