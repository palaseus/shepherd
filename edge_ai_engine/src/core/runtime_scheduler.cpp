/**
 * @file runtime_scheduler.cpp
 * @brief Runtime task scheduling and execution management implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "core/runtime_scheduler.h"
#include "profiling/profiler.h"
#include <stdexcept>
#include <queue>
#include <condition_variable>
#include <algorithm>
#include <thread>

namespace edge_ai {

RuntimeScheduler::RuntimeScheduler(const SchedulerConfig& config)
    : config_(config)
    , initialized_(false)
    , profiling_enabled_(false)
    , shutdown_requested_(false)
    , next_task_id_(1) {
}

RuntimeScheduler::~RuntimeScheduler() {
    Shutdown();
}

Status RuntimeScheduler::Initialize() {
    try {
        if (initialized_) {
            return Status::ALREADY_INITIALIZED;
        }
        
        // Initialize worker threads
        int num_threads = config_.num_worker_threads;
        if (num_threads <= 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        
        for (int i = 0; i < num_threads; ++i) {
            worker_threads_.emplace_back(&RuntimeScheduler::WorkerThread, this, i);
        }
        
        initialized_ = true;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status RuntimeScheduler::Shutdown() {
    try {
        if (!initialized_) {
            return Status::SUCCESS;
        }
        
        shutdown_requested_ = true;
        queue_cv_.notify_all();
        
        // Wait for worker threads to finish
        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        worker_threads_.clear();
        
        Cleanup();
        initialized_ = false;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status RuntimeScheduler::SubmitTask(std::shared_ptr<Task> task) {
    try {
        if (!initialized_) {
            return Status::NOT_INITIALIZED;
        }
        
        if (!task) {
            return Status::INVALID_ARGUMENT;
        }
        
        // Mark task enqueue event
        PROFILER_MARK_EVENT(0, "task_enqueue");
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.push(task);
            
            // Update statistics
            stats_.total_tasks_submitted.fetch_add(1);
            stats_.current_queue_size.store(task_queue_.size());
            
            // Update max queue size
            size_t current_size = task_queue_.size();
            size_t max_size = stats_.max_queue_size.load();
            while (current_size > max_size && !stats_.max_queue_size.compare_exchange_weak(max_size, current_size)) {
                // Retry if another thread updated max_queue_size
            }
        }
        
        queue_cv_.notify_one();
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status RuntimeScheduler::SubmitTask(std::shared_ptr<Task> task, TaskPriority priority) {
    try {
        if (!initialized_) {
            return Status::NOT_INITIALIZED;
        }
        
        if (!task) {
            return Status::INVALID_ARGUMENT;
        }
        
        // Set the task priority
        task->SetPriority(priority);
        
        return SubmitTask(task);
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status RuntimeScheduler::SubmitTask(std::shared_ptr<Task> task,
                                     [[maybe_unused]] std::function<void(Status, std::shared_ptr<Task>)> callback) {
    // For now, just submit the task normally
    // In practice, this would store the callback and call it when the task completes
    return SubmitTask(task);
}

Status RuntimeScheduler::WaitForTask([[maybe_unused]] uint64_t task_id, [[maybe_unused]] std::chrono::milliseconds timeout) {
    // Placeholder implementation
    return Status::SUCCESS;
}

Status RuntimeScheduler::CancelTask([[maybe_unused]] uint64_t task_id) {
    // Placeholder implementation
    return Status::SUCCESS;
}

TaskStatus RuntimeScheduler::GetTaskStatus([[maybe_unused]] uint64_t task_id) const {
    // Placeholder implementation
    return TaskStatus::COMPLETED;
}

SchedulerStats::Snapshot RuntimeScheduler::GetSchedulerStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_.GetSnapshot();
}

Status RuntimeScheduler::SetSchedulerConfig(const SchedulerConfig& config) {
    try {
        config_ = config;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

SchedulerConfig RuntimeScheduler::GetSchedulerConfig() const {
    return config_;
}

Status RuntimeScheduler::AddDevice(std::shared_ptr<Device> device) {
    try {
        if (!device) {
            return Status::INVALID_ARGUMENT;
        }
        
        std::lock_guard<std::mutex> lock(devices_mutex_);
        devices_.push_back(device);
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status RuntimeScheduler::RemoveDevice([[maybe_unused]] uint64_t device_id) {
    // Placeholder implementation
    return Status::SUCCESS;
}

std::vector<std::shared_ptr<Device>> RuntimeScheduler::GetAvailableDevices() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(devices_mutex_));
    return devices_;
}

Status RuntimeScheduler::SetTaskAffinity([[maybe_unused]] uint64_t task_id, [[maybe_unused]] uint64_t device_id) {
    // Placeholder implementation
    return Status::SUCCESS;
}

void RuntimeScheduler::SetProfiling(bool enable) {
    profiling_enabled_ = enable;
}

bool RuntimeScheduler::IsProfilingEnabled() const {
    return profiling_enabled_;
}

// Private methods
void RuntimeScheduler::WorkerThread([[maybe_unused]] int thread_id) {
    while (!shutdown_requested_) {
        std::shared_ptr<Task> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !task_queue_.empty() || shutdown_requested_; });
            
            if (shutdown_requested_) {
                break;
            }
            
            if (!task_queue_.empty()) {
                task = task_queue_.top();  // Get highest priority task
                task_queue_.pop();
                
                // Update statistics
                stats_.current_queue_size.store(task_queue_.size());
            }
        }
        
        if (task) {
            // Mark task dequeue and start execution
            PROFILER_MARK_EVENT(0, "task_dequeue");
            PROFILER_SCOPED_EVENT(0, "task_execute");
            
            auto start_time = std::chrono::high_resolution_clock::now();
            Status status = ExecuteTask(task);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            UpdateStats(task, execution_time, status == Status::SUCCESS);
        }
    }
}

Status RuntimeScheduler::ExecuteTask(std::shared_ptr<Task> task) {
    try {
        if (!task) {
            return Status::INVALID_ARGUMENT;
        }
        
        task->SetStatus(TaskStatus::RUNNING);
        
        // Simulate task execution based on task type
        Status status = Status::SUCCESS;
        switch (task->GetType()) {
            case TaskType::INFERENCE:
                // Simulate inference execution time
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                break;
            case TaskType::OPTIMIZATION:
                // Simulate optimization execution time
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                break;
            case TaskType::MEMORY_ALLOCATION:
                // Simulate memory allocation time
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                break;
            case TaskType::DATA_PREPROCESSING:
            case TaskType::DATA_POSTPROCESSING:
                // Simulate data processing time
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                break;
            default:
                status = Status::INVALID_ARGUMENT;
                break;
        }
        
        if (status == Status::SUCCESS) {
            task->SetStatus(TaskStatus::COMPLETED);
        } else {
            task->SetStatus(TaskStatus::FAILED);
        }
        
        return status;
    } catch (const std::exception& e) {
        if (task) {
            task->SetStatus(TaskStatus::FAILED);
        }
        return Status::FAILURE;
    }
}

std::shared_ptr<Device> RuntimeScheduler::SelectDevice([[maybe_unused]] std::shared_ptr<Task> task) {
    // Placeholder implementation - just return the first available device
    std::lock_guard<std::mutex> lock(devices_mutex_);
    if (!devices_.empty()) {
        return devices_[0];
    }
    return nullptr;
}

void RuntimeScheduler::UpdateStats([[maybe_unused]] std::shared_ptr<Task> task, std::chrono::microseconds execution_time, bool success) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (success) {
        stats_.total_tasks_completed.fetch_add(1);
    } else {
        stats_.total_tasks_failed.fetch_add(1);
    }
    
    // Update execution time statistics
    auto current_total = stats_.total_execution_time.load();
    stats_.total_execution_time.store(current_total + execution_time);
    
    auto current_min = stats_.min_execution_time.load();
    if (execution_time < current_min) {
        stats_.min_execution_time.store(execution_time);
    }
    
    auto current_max = stats_.max_execution_time.load();
    if (execution_time > current_max) {
        stats_.max_execution_time.store(execution_time);
    }
}

void RuntimeScheduler::Cleanup() {
    // Clear task queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!task_queue_.empty()) {
            task_queue_.pop();
        }
    }
    
    // Clear active tasks
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        active_tasks_.clear();
    }
    
    // Clear devices
    {
        std::lock_guard<std::mutex> lock(devices_mutex_);
        devices_.clear();
    }
}


} // namespace edge_ai
