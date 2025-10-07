/**
 * @file error_handler.cpp
 * @brief Error handling utility implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "utils/error_handler.h"
#include "utils/logger.h"
#include <stdexcept>
#include <sstream>

namespace EdgeAI {

ErrorHandler::ErrorHandler() = default;

ErrorHandler::~ErrorHandler() = default;

void ErrorHandler::HandleError(edge_ai::Status status, const std::string& message) {
    std::string error_message = "Error: " + StatusToString(status) + " - " + message;
    
    // Log the error
    Logger::GetInstance().Error(error_message);
    
    // In debug mode, throw an exception
    #ifdef DEBUG
    throw std::runtime_error(error_message);
    #endif
}

void ErrorHandler::HandleException(const std::exception& e) {
    std::string error_message = "Exception: " + std::string(e.what());
    
    // Log the error
    Logger::GetInstance().Error(error_message);
    
    // In debug mode, re-throw the exception
    #ifdef DEBUG
    throw;
    #endif
}

std::string ErrorHandler::StatusToString(edge_ai::Status status) {
    switch (status) {
        case edge_ai::Status::SUCCESS: return "SUCCESS";
        case edge_ai::Status::FAILURE: return "FAILURE";
        case edge_ai::Status::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case edge_ai::Status::OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case edge_ai::Status::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
        case edge_ai::Status::NOT_INITIALIZED: return "NOT_INITIALIZED";
        case edge_ai::Status::ALREADY_INITIALIZED: return "ALREADY_INITIALIZED";
        case edge_ai::Status::MODEL_NOT_LOADED: return "MODEL_NOT_LOADED";
        case edge_ai::Status::MODEL_ALREADY_LOADED: return "MODEL_ALREADY_LOADED";
        case edge_ai::Status::OPTIMIZATION_FAILED: return "OPTIMIZATION_FAILED";
        case edge_ai::Status::INFERENCE_FAILED: return "INFERENCE_FAILED";
        case edge_ai::Status::HARDWARE_NOT_AVAILABLE: return "HARDWARE_NOT_AVAILABLE";
        case edge_ai::Status::INVALID_MODEL_FORMAT: return "INVALID_MODEL_FORMAT";
        case edge_ai::Status::UNSUPPORTED_OPERATION: return "UNSUPPORTED_OPERATION";
        default: return "UNKNOWN_STATUS";
    }
}

} // namespace EdgeAI
