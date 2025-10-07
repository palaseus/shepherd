/**
 * @file error_handler.h
 * @brief Error handling utility interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the ErrorHandler class for error handling in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <string>
#include <exception>

namespace EdgeAI {

/**
 * @class ErrorHandler
 * @brief Error handling utility class
 * 
 * The ErrorHandler class provides centralized error handling for the Edge AI Engine.
 */
class ErrorHandler {
public:
    /**
     * @brief Constructor
     */
    ErrorHandler();
    
    /**
     * @brief Destructor
     */
    ~ErrorHandler();
    
    // Disable copy constructor and assignment operator
    ErrorHandler(const ErrorHandler&) = delete;
    ErrorHandler& operator=(const ErrorHandler&) = delete;
    
    /**
     * @brief Handle an error
     * @param status Error status
     * @param message Error message
     */
    void HandleError(edge_ai::Status status, const std::string& message);
    
    /**
     * @brief Handle an exception
     * @param e Exception to handle
     */
    void HandleException(const std::exception& e);
    
    /**
     * @brief Convert status to string
     * @param status Status to convert
     * @return String representation of status
     */
    std::string StatusToString(edge_ai::Status status);
};

} // namespace EdgeAI
