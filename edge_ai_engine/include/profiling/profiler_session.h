/**
 * @file profiler_session.h
 * @brief Profiler session management for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the ProfilerSession class which manages individual
 * profiling sessions and provides session-level statistics and export capabilities.
 */

#pragma once

#include "../core/types.h"
#include "profiler.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>

namespace edge_ai {

/**
 * @struct SessionStats
 * @brief Statistics for a profiling session
 */
struct SessionStats {
    std::atomic<uint64_t> total_events{0};
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> total_duration_ns{0};
    std::atomic<uint64_t> min_duration_ns{std::numeric_limits<uint64_t>::max()};
    std::atomic<uint64_t> max_duration_ns{0};
    std::atomic<uint64_t> overflow_count{0};
    
    SessionStats() = default;
    
    // Non-atomic version for return values
    struct Snapshot {
        uint64_t total_events;
        uint64_t total_requests;
        uint64_t total_duration_ns;
        uint64_t min_duration_ns;
        uint64_t max_duration_ns;
        uint64_t overflow_count;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_events = total_events.load();
        snapshot.total_requests = total_requests.load();
        snapshot.total_duration_ns = total_duration_ns.load();
        snapshot.min_duration_ns = min_duration_ns.load();
        snapshot.max_duration_ns = max_duration_ns.load();
        snapshot.overflow_count = overflow_count.load();
        return snapshot;
    }
};

/**
 * @class ProfilerSession
 * @brief Individual profiling session
 * 
 * The ProfilerSession class manages a single profiling session, collecting
 * events from multiple threads and providing session-level statistics.
 */
class ProfilerSession {
public:
    /**
     * @brief Constructor
     * @param session_name Name of the session
     */
    explicit ProfilerSession(const std::string& session_name);
    
    /**
     * @brief Destructor
     */
    ~ProfilerSession() = default;
    
    // Disable copy constructor and assignment operator
    ProfilerSession(const ProfilerSession&) = delete;
    ProfilerSession& operator=(const ProfilerSession&) = delete;
    
    /**
     * @brief Start the session
     * @return Status indicating success or failure
     */
    Status Start();
    
    /**
     * @brief Stop the session
     * @return Status indicating success or failure
     */
    Status Stop();
    
    /**
     * @brief Add events to the session
     * @param events Vector of events to add
     */
    void AddEvents(const std::vector<ProfilerEvent>& events);
    
    /**
     * @brief Add a single event to the session
     * @param event Event to add
     */
    void AddEvent(const ProfilerEvent& event);
    
    /**
     * @brief Get session metadata
     * @return Session metadata
     */
    SessionMetadata GetMetadata() const;
    
    /**
     * @brief Get session statistics
     * @return Session statistics snapshot
     */
    SessionStats::Snapshot GetStats() const;
    
    /**
     * @brief Get all events in the session
     * @return Vector of all events
     */
    std::vector<ProfilerEvent> GetAllEvents() const;
    
    /**
     * @brief Get events for a specific request
     * @param request_id Request identifier
     * @return Vector of events for the request
     */
    std::vector<ProfilerEvent> GetEventsForRequest(uint64_t request_id) const;
    
    /**
     * @brief Check if session is active
     * @return True if session is active
     */
    bool IsActive() const { return active_; }
    
    /**
     * @brief Get session name
     * @return Session name
     */
    const std::string& GetName() const { return metadata_.session_name; }
    
    /**
     * @brief Set session metadata
     * @param key Metadata key
     * @param value Metadata value
     */
    void SetMetadata(const std::string& key, const std::string& value);
    
    /**
     * @brief Get session metadata value
     * @param key Metadata key
     * @return Metadata value (empty if not found)
     */
    std::string GetMetadata(const std::string& key) const;
    
    /**
     * @brief Clear all events and reset session
     */
    void Clear();
    
    /**
     * @brief Export session to JSON format
     * @param file_path Output file path
     * @return Status indicating success or failure
     */
    Status ExportToJson(const std::string& file_path) const;

private:
    /**
     * @brief Update session statistics
     * @param event Event to process for statistics
     */
    void UpdateStats(const ProfilerEvent& event);
    
    /**
     * @brief Generate JSON content for export
     * @return JSON string
     */
    std::string GenerateJsonContent() const;

private:
    std::string session_name_;
    SessionMetadata metadata_;
    SessionStats stats_;
    std::atomic<bool> active_{false};
    
    // Event storage
    mutable std::mutex events_mutex_;
    std::vector<ProfilerEvent> events_;
    std::unordered_map<uint64_t, std::vector<size_t>> request_event_indices_;
    
    // Custom metadata
    std::unordered_map<std::string, std::string> custom_metadata_;
    mutable std::mutex metadata_mutex_;
};

} // namespace edge_ai