/**
 * @file task_priority.h
 * @brief Task priority interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the TaskPriority enum for task priorities in the Edge AI Engine.
 */

#pragma once

namespace edge_ai {

/**
 * @enum TaskPriority
 * @brief Task priorities
 */
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

} // namespace edge_ai
