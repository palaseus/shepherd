/**
 * @file performance_counter.h
 * @brief Performance counter interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the PerformanceCounter class for performance counting in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <string>

namespace edge_ai {

/**
 * @class PerformanceCounter
 * @brief Performance counter utility class
 * 
 * The PerformanceCounter class provides functionality for counting and
 * measuring performance metrics in the Edge AI Engine.
 */
class PerformanceCounter {
public:
    /**
     * @brief Constructor
     */
    PerformanceCounter();
    
    /**
     * @brief Destructor
     */
    ~PerformanceCounter();
    
    // Disable copy constructor and assignment operator
    PerformanceCounter(const PerformanceCounter&) = delete;
    PerformanceCounter& operator=(const PerformanceCounter&) = delete;
    
    /**
     * @brief Initialize the performance counter
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the performance counter
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Start a counter
     * @param name Name of the counter
     * @return Status indicating success or failure
     */
    Status StartCounter(const std::string& name);
    
    /**
     * @brief Stop a counter
     * @param name Name of the counter
     * @return Status indicating success or failure
     */
    Status StopCounter(const std::string& name);
};

} // namespace edge_ai
