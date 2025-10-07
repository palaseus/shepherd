/**
 * @file request_priority.h
 * @brief Request priority interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the RequestPriority enum for request priorities in the Edge AI Engine.
 */

#pragma once

namespace edge_ai {

/**
 * @enum RequestPriority
 * @brief Request priorities
 */
enum class RequestPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

} // namespace edge_ai
