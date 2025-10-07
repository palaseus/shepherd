/**
 * @file metrics_collector.h
 * @brief Metrics collector interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MetricsCollector class for collecting metrics in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"

namespace edge_ai {

/**
 * @class MetricsCollector
 * @brief Metrics collector utility class
 * 
 * The MetricsCollector class provides functionality for collecting and
 * aggregating performance metrics in the Edge AI Engine.
 */
class MetricsCollector {
public:
    /**
     * @brief Constructor
     */
    MetricsCollector();
    
    /**
     * @brief Destructor
     */
    ~MetricsCollector();
    
    // Disable copy constructor and assignment operator
    MetricsCollector(const MetricsCollector&) = delete;
    MetricsCollector& operator=(const MetricsCollector&) = delete;
    
    /**
     * @brief Initialize the metrics collector
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the metrics collector
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Collect metrics
     * @return Status indicating success or failure
     */
    Status CollectMetrics();
};

} // namespace edge_ai
