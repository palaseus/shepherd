/**
 * @file task_status.h
 * @brief Task status interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the TaskStatus enum for task statuses in the Edge AI Engine.
 */

#pragma once

namespace edge_ai {

/**
 * @enum TaskStatus
 * @brief Task status
 */
enum class TaskStatus {
    PENDING = 0,
    RUNNING = 1,
    COMPLETED = 2,
    FAILED = 3,
    CANCELLED = 4
};

} // namespace edge_ai
