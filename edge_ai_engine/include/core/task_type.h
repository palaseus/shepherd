/**
 * @file task_type.h
 * @brief Task type interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the TaskType enum for task types in the Edge AI Engine.
 */

#pragma once

namespace edge_ai {

/**
 * @enum TaskType
 * @brief Types of tasks
 */
enum class TaskType {
    INFERENCE = 0,
    OPTIMIZATION = 1,
    MEMORY_ALLOCATION = 2,
    DATA_PROCESSING = 3,
    PROFILING = 4,
    CLEANUP = 5
};

} // namespace edge_ai
