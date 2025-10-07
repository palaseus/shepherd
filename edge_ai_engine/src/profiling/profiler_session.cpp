/**
 * @file profiler_session.cpp
 * @brief Profiler session management implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "profiling/profiler_session.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>

namespace edge_ai {

ProfilerSession::ProfilerSession(const std::string& session_name)
    : session_name_(session_name) {
    metadata_.session_name = session_name;
}

Status ProfilerSession::Start() {
    if (active_.load()) {
        return Status::ALREADY_INITIALIZED;
    }
    
    metadata_.start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    metadata_.end_time_ns = 0;
    metadata_.request_count = 0;
    metadata_.overflow_detected = false;
    metadata_.total_events = 0;
    
    active_.store(true);
    return Status::SUCCESS;
}

Status ProfilerSession::Stop() {
    if (!active_.load()) {
        return Status::FAILURE;
    }
    
    metadata_.end_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    active_.store(false);
    return Status::SUCCESS;
}

void ProfilerSession::AddEvents(const std::vector<ProfilerEvent>& events) {
    if (events.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    size_t start_index = events_.size();
    events_.insert(events_.end(), events.begin(), events.end());
    
    // Update request indices
    for (size_t i = 0; i < events.size(); ++i) {
        const ProfilerEvent& event = events[start_index + i];
        if (event.request_id != 0) {
            request_event_indices_[event.request_id].push_back(start_index + i);
        }
        UpdateStats(event);
    }
    
    metadata_.total_events = events_.size();
}

void ProfilerSession::AddEvent(const ProfilerEvent& event) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    size_t event_index = events_.size();
    events_.push_back(event);
    
    // Update request indices
    if (event.request_id != 0) {
        request_event_indices_[event.request_id].push_back(event_index);
    }
    
    UpdateStats(event);
    metadata_.total_events = events_.size();
}

SessionMetadata ProfilerSession::GetMetadata() const {
    return metadata_;
}

SessionStats::Snapshot ProfilerSession::GetStats() const {
    return stats_.GetSnapshot();
}

std::vector<ProfilerEvent> ProfilerSession::GetAllEvents() const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    return events_;
}

std::vector<ProfilerEvent> ProfilerSession::GetEventsForRequest(uint64_t request_id) const {
    std::lock_guard<std::mutex> lock(events_mutex_);
    
    std::vector<ProfilerEvent> result;
    auto it = request_event_indices_.find(request_id);
    if (it != request_event_indices_.end()) {
        result.reserve(it->second.size());
        for (size_t index : it->second) {
            if (index < events_.size()) {
                result.push_back(events_[index]);
            }
        }
    }
    
    return result;
}

void ProfilerSession::SetMetadata(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    custom_metadata_[key] = value;
}

std::string ProfilerSession::GetMetadata(const std::string& key) const {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    auto it = custom_metadata_.find(key);
    if (it != custom_metadata_.end()) {
        return it->second;
    }
    return "";
}

void ProfilerSession::Clear() {
    std::lock_guard<std::mutex> events_lock(events_mutex_);
    std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
    
    events_.clear();
    request_event_indices_.clear();
    custom_metadata_.clear();
    
    // Reset stats
    stats_.total_events.store(0);
    stats_.total_requests.store(0);
    stats_.total_duration_ns.store(0);
    stats_.min_duration_ns.store(std::numeric_limits<uint64_t>::max());
    stats_.max_duration_ns.store(0);
    stats_.overflow_count.store(0);
    
    // Reset metadata
    metadata_.total_events = 0;
    metadata_.request_count = 0;
    metadata_.overflow_detected = false;
}

Status ProfilerSession::ExportToJson(const std::string& file_path) const {
    try {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            return Status::FAILURE;
        }
        
        std::string json_content = GenerateJsonContent();
        file << json_content;
        file.close();
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

void ProfilerSession::UpdateStats(const ProfilerEvent& event) {
    stats_.total_events.fetch_add(1);
    
    if (event.request_id != 0) {
        // Count unique requests
        static thread_local std::unordered_set<uint64_t> seen_requests;
        if (seen_requests.find(event.request_id) == seen_requests.end()) {
            seen_requests.insert(event.request_id);
            stats_.total_requests.fetch_add(1);
        }
    }
    
    if (event.event_type == EventType::SCOPED_END && event.duration_ns > 0) {
        stats_.total_duration_ns.fetch_add(event.duration_ns);
        
        // Update min/max duration
        uint64_t current_min = stats_.min_duration_ns.load();
        while (event.duration_ns < current_min && 
               !stats_.min_duration_ns.compare_exchange_weak(current_min, event.duration_ns)) {
            // Retry if compare_exchange_weak failed
        }
        
        uint64_t current_max = stats_.max_duration_ns.load();
        while (event.duration_ns > current_max && 
               !stats_.max_duration_ns.compare_exchange_weak(current_max, event.duration_ns)) {
            // Retry if compare_exchange_weak failed
        }
    }
}

std::string ProfilerSession::GenerateJsonContent() const {
    std::ostringstream json;
    
    // Get events and metadata
    std::vector<ProfilerEvent> events = GetAllEvents();
    SessionMetadata metadata = GetMetadata();
    SessionStats::Snapshot stats = GetStats();
    
    json << "{\n";
    
    // Session metadata
    json << "  \"session_metadata\": {\n";
    json << "    \"session_name\": \"" << metadata.session_name << "\",\n";
    json << "    \"start_time_ns\": " << metadata.start_time_ns << ",\n";
    json << "    \"end_time_ns\": " << metadata.end_time_ns << ",\n";
    json << "    \"request_count\": " << metadata.request_count << ",\n";
    json << "    \"overflow_detected\": " << (metadata.overflow_detected ? "true" : "false") << ",\n";
    json << "    \"backend_type\": \"" << metadata.backend_type << "\",\n";
    json << "    \"model_id\": \"" << metadata.model_id << "\",\n";
    json << "    \"total_events\": " << metadata.total_events << "\n";
    json << "  },\n";
    
    // Session statistics
    json << "  \"session_stats\": {\n";
    json << "    \"total_events\": " << stats.total_events << ",\n";
    json << "    \"total_requests\": " << stats.total_requests << ",\n";
    json << "    \"total_duration_ns\": " << stats.total_duration_ns << ",\n";
    json << "    \"min_duration_ns\": " << stats.min_duration_ns << ",\n";
    json << "    \"max_duration_ns\": " << stats.max_duration_ns << ",\n";
    json << "    \"overflow_count\": " << stats.overflow_count << "\n";
    json << "  },\n";
    
    // Events array
    json << "  \"events\": [\n";
    for (size_t i = 0; i < events.size(); ++i) {
        const ProfilerEvent& event = events[i];
        
        json << "    {\n";
        json << "      \"timestamp_ns\": " << event.timestamp_ns << ",\n";
        json << "      \"thread_id\": \"" << event.thread_id << "\",\n";
        json << "      \"request_id\": " << event.request_id << ",\n";
        json << "      \"event_type\": " << static_cast<int>(event.event_type) << ",\n";
        json << "      \"event_name\": \"" << event.event_name << "\",\n";
        json << "      \"duration_ns\": " << event.duration_ns << "\n";
        json << "    }";
        
        if (i < events.size() - 1) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    
    json << "}\n";
    
    return json.str();
}

} // namespace edge_ai