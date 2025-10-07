/**
 * @file profiler.h
 * @brief High-performance profiling system for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the Profiler class which provides low-overhead, thread-safe
 * profiling capabilities with high-resolution timing and JSON export functionality.
 */

#pragma once

#include "../core/types.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <fstream>
#include <unordered_set>

namespace edge_ai {

// Forward declarations
class ProfilerSession;

/**
 * @enum EventType
 * @brief Types of profiling events
 */
enum class EventType {
    MARK = 0,           // Discrete timestamped mark
    SCOPED_START = 1,   // Start of scoped event
    SCOPED_END = 2      // End of scoped event
};

/**
 * @struct ProfilerEvent
 * @brief Individual profiling event record
 */
struct ProfilerEvent {
    uint64_t timestamp_ns;          // High-resolution timestamp
    std::thread::id thread_id;      // Thread that recorded the event
    uint64_t request_id;            // Request ID (0 for global events)
    EventType event_type;           // Type of event
    std::string event_name;         // Event name/identifier
    uint64_t duration_ns;           // Duration for scoped events (0 for marks)
    
    ProfilerEvent() = default;
    ProfilerEvent(uint64_t ts, std::thread::id tid, uint64_t rid, 
                  EventType type, const std::string& name, uint64_t dur = 0)
        : timestamp_ns(ts), thread_id(tid), request_id(rid), 
          event_type(type), event_name(name), duration_ns(dur) {}
};

/**
 * @struct SessionMetadata
 * @brief Lightweight metadata for profiling sessions
 */
struct SessionMetadata {
    std::string session_name;       // Session identifier
    uint64_t start_time_ns;         // Session start timestamp
    uint64_t end_time_ns;           // Session end timestamp (0 if active)
    uint64_t request_count;         // Number of requests in session
    bool overflow_detected;         // True if thread buffers overflowed
    std::string backend_type;       // Primary backend used
    std::string model_id;           // Model identifier
    size_t total_events;            // Total events recorded
    
    SessionMetadata() : start_time_ns(0), end_time_ns(0), request_count(0),
                       overflow_detected(false), total_events(0) {}
};

/**
 * @class ThreadLocalBuffer
 * @brief Thread-local event buffer for low-overhead profiling
 */
class ThreadLocalBuffer {
public:
    static constexpr size_t BUFFER_SIZE = 1024;  // Fixed-size ring buffer
    
    ThreadLocalBuffer();
    ~ThreadLocalBuffer();
    
    /**
     * @brief Add event to thread-local buffer
     * @param event Event to add
     * @return True if added successfully, false if buffer full
     */
    bool AddEvent(const ProfilerEvent& event);
    
    /**
     * @brief Extract all events from buffer (clears buffer)
     * @return Vector of events
     */
    std::vector<ProfilerEvent> ExtractEvents();
    
    /**
     * @brief Check if buffer has overflowed
     * @return True if overflow detected
     */
    bool HasOverflowed() const { return overflow_detected_; }
    
    /**
     * @brief Get current buffer size
     * @return Number of events in buffer
     */
    size_t GetSize() const { return size_; }

private:
    ProfilerEvent events_[BUFFER_SIZE];
    std::atomic<size_t> write_index_{0};
    std::atomic<size_t> size_{0};
    std::atomic<bool> overflow_detected_{false};
};

/**
 * @class Profiler
 * @brief High-performance profiling system
 * 
 * The Profiler class provides low-overhead, thread-safe profiling with:
 * - Thread-local event buffers to minimize contention
 * - High-resolution timing using steady_clock
 * - Per-request and global session support
 * - JSON export functionality
 * - RAII scoped event support
 */
class Profiler {
public:
    /**
     * @brief Get singleton instance
     * @return Profiler instance
     */
    static Profiler& GetInstance();
    
    /**
     * @brief Initialize profiler
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown profiler
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Register an event source
     * @param name Event source name
     * @return Status indicating success or failure
     */
    Status RegisterEventSource(const std::string& name);
    
    /**
     * @brief Start a global profiling session
     * @param session_name Name of the session
     * @return Status indicating success or failure
     */
    Status StartGlobalSession(const std::string& session_name);
    
    /**
     * @brief Stop the current global session
     * @return Status indicating success or failure
     */
    Status StopGlobalSession();
    
    /**
     * @brief Start a session for a specific request
     * @param request_id Request identifier
     * @return Status indicating success or failure
     */
    Status StartSessionForRequest(uint64_t request_id);
    
    /**
     * @brief End session for a specific request
     * @param request_id Request identifier
     * @return Status indicating success or failure
     */
    Status EndSessionForRequest(uint64_t request_id);
    
    /**
     * @brief Mark a discrete event
     * @param request_id Request identifier (0 for global events)
     * @param event_name Event name
     * @return Status indicating success or failure
     */
    Status MarkEvent(uint64_t request_id, const std::string& event_name);
    
    /**
     * @brief Create a scoped event (RAII)
     * @param request_id Request identifier (0 for global events)
     * @param event_name Event name
     * @return Unique pointer to scoped event
     */
    std::unique_ptr<class ScopedEvent> CreateScopedEvent(uint64_t request_id, 
                                                         const std::string& event_name);
    
    /**
     * @brief Export session as JSON
     * @param session_name Session name to export
     * @param file_path Output file path
     * @return Status indicating success or failure
     */
    Status ExportSessionAsJson(const std::string& session_name, const std::string& file_path);
    
    /**
     * @brief Get session metadata
     * @param session_name Session name
     * @return Session metadata (empty if not found)
     */
    SessionMetadata GetSessionMetadata(const std::string& session_name) const;
    
    /**
     * @brief Check if profiler is enabled
     * @return True if profiler is enabled
     */
    bool IsEnabled() const { return enabled_; }
    
    /**
     * @brief Enable or disable profiler
     * @param enabled Enable flag
     */
    void SetEnabled(bool enabled) { enabled_ = enabled; }

public:
    Profiler() = default;
    ~Profiler() = default;
    
    // Disable copy constructor and assignment operator
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    
    /**
     * @brief Get thread-local buffer for current thread
     * @return Thread-local buffer
     */
    ThreadLocalBuffer* GetThreadLocalBuffer();
    
    /**
     * @brief Flush all thread-local buffers to global storage
     */
    void FlushAllThreadBuffers();
    
    /**
     * @brief Get current high-resolution timestamp
     * @return Timestamp in nanoseconds
     */
    uint64_t GetCurrentTimestamp() const;
    
    /**
     * @brief Add event to appropriate buffer
     * @param event Event to add
     */
    void AddEventInternal(const ProfilerEvent& event);

private:
    bool initialized_{false};
    bool enabled_{true};
    
    // Thread-local storage
    thread_local static ThreadLocalBuffer* thread_buffer_;
    
    // Global storage
    mutable std::mutex global_mutex_;
    std::unordered_map<std::string, std::shared_ptr<ProfilerSession>> sessions_;
    std::string current_global_session_;
    
    // Event sources
    std::unordered_set<std::string> event_sources_;
    mutable std::mutex sources_mutex_;
};

/**
 * @class ScopedEvent
 * @brief RAII wrapper for scoped profiling events
 */
class ScopedEvent {
public:
    /**
     * @brief Constructor - records start event
     * @param request_id Request identifier
     * @param event_name Event name
     */
    ScopedEvent(uint64_t request_id, const std::string& event_name);
    
    /**
     * @brief Destructor - records end event
     */
    ~ScopedEvent();
    
    // Disable copy constructor and assignment operator
    ScopedEvent(const ScopedEvent&) = delete;
    ScopedEvent& operator=(const ScopedEvent&) = delete;
    
    // Move constructor and assignment operator
    ScopedEvent(ScopedEvent&& other) noexcept;
    ScopedEvent& operator=(ScopedEvent&& other) noexcept;

private:
    uint64_t request_id_;
    std::string event_name_;
    uint64_t start_time_ns_;
    bool active_;
};

// Convenience macros for profiling
#ifdef PROFILER_ENABLED
    #define PROFILER_SCOPED_EVENT(request_id, event_name) \
        auto _profiler_scoped_event = Profiler::GetInstance().CreateScopedEvent(request_id, event_name)
    
    #define PROFILER_MARK_EVENT(request_id, event_name) \
        Profiler::GetInstance().MarkEvent(request_id, event_name)
#else
    #define PROFILER_SCOPED_EVENT(request_id, event_name) \
        do { (void)(request_id); (void)(event_name); } while(0)
    
    #define PROFILER_MARK_EVENT(request_id, event_name) \
        do { (void)(request_id); (void)(event_name); } while(0)
#endif

} // namespace edge_ai