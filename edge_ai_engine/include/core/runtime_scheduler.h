/**
 * @file runtime_scheduler.h
 * @brief Runtime task scheduling and execution management
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the RuntimeScheduler class which handles task scheduling,
 * resource management, and execution optimization for the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>
#include <unordered_map>

namespace edge_ai {

// Forward declarations
class Task;
class Device;
class Profiler;

/**
 * @class RuntimeScheduler
 * @brief Runtime task scheduling and execution management
 * 
 * The RuntimeScheduler class manages task execution, resource allocation,
 * and performance optimization for the Edge AI Engine.
 */
class RuntimeScheduler {
public:
    /**
     * @brief Constructor
     * @param config Scheduler configuration
     */
    explicit RuntimeScheduler(const SchedulerConfig& config = SchedulerConfig{});
    
    /**
     * @brief Destructor
     */
    ~RuntimeScheduler();
    
    // Disable copy constructor and assignment operator
    RuntimeScheduler(const RuntimeScheduler&) = delete;
    RuntimeScheduler& operator=(const RuntimeScheduler&) = delete;
    
    /**
     * @brief Initialize the scheduler
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the scheduler
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Submit a task for execution
     * @param task Task to execute
     * @return Status indicating success or failure
     */
    Status SubmitTask(std::shared_ptr<Task> task);
    
    /**
     * @brief Submit a task with priority
     * @param task Task to execute
     * @param priority Task priority
     * @return Status indicating success or failure
     */
    Status SubmitTask(std::shared_ptr<Task> task, TaskPriority priority);
    
    /**
     * @brief Submit a task with callback
     * @param task Task to execute
     * @param callback Callback function for completion
     * @return Status indicating success or failure
     */
    Status SubmitTask(std::shared_ptr<Task> task,
                     std::function<void(Status, std::shared_ptr<Task>)> callback);
    
    /**
     * @brief Wait for task completion
     * @param task_id Task ID to wait for
     * @param timeout Maximum time to wait
     * @return Status indicating success or failure
     */
    Status WaitForTask(uint64_t task_id, std::chrono::milliseconds timeout = std::chrono::milliseconds::max());
    
    /**
     * @brief Cancel a task
     * @param task_id Task ID to cancel
     * @return Status indicating success or failure
     */
    Status CancelTask(uint64_t task_id);
    
    /**
     * @brief Get task status
     * @param task_id Task ID
     * @return Task status
     */
    TaskStatus GetTaskStatus(uint64_t task_id) const;
    
    /**
     * @brief Get scheduler statistics
     * @return Scheduler statistics
     */
    SchedulerStats::Snapshot GetSchedulerStats() const;
    
    /**
     * @brief Set scheduler configuration
     * @param config Scheduler configuration
     * @return Status indicating success or failure
     */
    Status SetSchedulerConfig(const SchedulerConfig& config);
    
    /**
     * @brief Get current scheduler configuration
     * @return Current scheduler configuration
     */
    SchedulerConfig GetSchedulerConfig() const;
    
    /**
     * @brief Add a device for task execution
     * @param device Device to add
     * @return Status indicating success or failure
     */
    Status AddDevice(std::shared_ptr<Device> device);
    
    /**
     * @brief Remove a device
     * @param device_id Device ID to remove
     * @return Status indicating success or failure
     */
    Status RemoveDevice(uint64_t device_id);
    
    /**
     * @brief Get available devices
     * @return Vector of available devices
     */
    std::vector<std::shared_ptr<Device>> GetAvailableDevices() const;
    
    /**
     * @brief Set task affinity
     * @param task_id Task ID
     * @param device_id Device ID for affinity
     * @return Status indicating success or failure
     */
    Status SetTaskAffinity(uint64_t task_id, uint64_t device_id);
    
    /**
     * @brief Enable or disable profiling
     * @param enable Enable profiling
     */
    void SetProfiling(bool enable);
    
    /**
     * @brief Check if profiling is enabled
     * @return True if profiling is enabled
     */
    bool IsProfilingEnabled() const;

private:
    // Configuration
    SchedulerConfig config_;
    
    // State
    bool initialized_;
    bool profiling_enabled_;
    std::atomic<bool> shutdown_requested_;
    
    // Threading
    std::vector<std::thread> worker_threads_;
    
    // Priority queue for tasks (higher priority first)
    struct TaskComparator {
        bool operator()(const std::shared_ptr<Task>& a, const std::shared_ptr<Task>& b) const {
            // Higher priority value means lower priority in the queue
            return static_cast<int>(a->GetPriority()) > static_cast<int>(b->GetPriority());
        }
    };
    std::priority_queue<std::shared_ptr<Task>, std::vector<std::shared_ptr<Task>>, TaskComparator> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Task management
    std::unordered_map<uint64_t, std::shared_ptr<Task>> active_tasks_;
    std::mutex tasks_mutex_;
    std::atomic<uint64_t> next_task_id_;
    
    // Device management
    std::vector<std::shared_ptr<Device>> devices_;
    std::mutex devices_mutex_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    SchedulerStats stats_;
    
    /**
     * @brief Worker thread function
     * @param thread_id Worker thread ID
     */
    void WorkerThread(int thread_id);
    
    /**
     * @brief Execute a task
     * @param task Task to execute
     * @return Status indicating success or failure
     */
    Status ExecuteTask(std::shared_ptr<Task> task);
    
    /**
     * @brief Select device for task execution
     * @param task Task to execute
     * @return Selected device
     */
    std::shared_ptr<Device> SelectDevice(std::shared_ptr<Task> task);
    
    /**
     * @brief Update scheduler statistics
     * @param task Task that was executed
     * @param execution_time Task execution time
     * @param success Whether task was successful
     */
    void UpdateStats(std::shared_ptr<Task> task, std::chrono::microseconds execution_time, bool success);
    
    /**
     * @brief Cleanup resources
     */
    void Cleanup();
};

} // namespace edge_ai
