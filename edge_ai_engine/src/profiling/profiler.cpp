/**
 * @file profiler.cpp
 * @brief High-performance profiling system implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "profiling/profiler.h"
#include "profiling/profiler_session.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace edge_ai {

// Thread-local buffer instance
thread_local ThreadLocalBuffer* Profiler::thread_buffer_ = nullptr;

// ThreadLocalBuffer implementation
ThreadLocalBuffer::ThreadLocalBuffer() = default;

ThreadLocalBuffer::~ThreadLocalBuffer() = default;

bool ThreadLocalBuffer::AddEvent(const ProfilerEvent& event) {
    size_t current_index = write_index_.load();
    size_t next_index = (current_index + 1) % BUFFER_SIZE;
    
    // Check if buffer is full
    if (size_.load() >= BUFFER_SIZE) {
        overflow_detected_.store(true);
        return false;
    }
    
    // Add event at current position
    events_[current_index] = event;
    write_index_.store(next_index);
    size_.fetch_add(1);
    
    return true;
}

std::vector<ProfilerEvent> ThreadLocalBuffer::ExtractEvents() {
    std::vector<ProfilerEvent> result;
    size_t current_size = size_.load();
    
    if (current_size == 0) {
        return result;
    }
    
    result.reserve(current_size);
    
    // Extract events in order
    size_t start_index = (write_index_.load() - current_size + BUFFER_SIZE) % BUFFER_SIZE;
    for (size_t i = 0; i < current_size; ++i) {
        size_t index = (start_index + i) % BUFFER_SIZE;
        result.push_back(events_[index]);
    }
    
    // Clear buffer
    size_.store(0);
    write_index_.store(0);
    
    return result;
}

// Profiler implementation
Profiler& Profiler::GetInstance() {
    static Profiler instance;
    return instance;
}

Status Profiler::Initialize() {
    if (initialized_) {
        return Status::ALREADY_INITIALIZED;
    }
    
    initialized_ = true;
    return Status::SUCCESS;
}

Status Profiler::Shutdown() {
    if (!initialized_) {
        return Status::SUCCESS;
    }
    
    // Flush all thread buffers before shutdown
    FlushAllThreadBuffers();
    
    initialized_ = false;
    return Status::SUCCESS;
}

Status Profiler::RegisterEventSource(const std::string& name) {
    if (name.empty()) {
        return Status::INVALID_ARGUMENT;
    }
    
    std::lock_guard<std::mutex> lock(sources_mutex_);
    event_sources_.insert(name);
    return Status::SUCCESS;
}

Status Profiler::StartGlobalSession(const std::string& session_name) {
    if (!initialized_ || !enabled_) {
        return Status::NOT_INITIALIZED;
    }
    
    if (session_name.empty()) {
        return Status::INVALID_ARGUMENT;
    }
    
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    // Stop current session if active (without acquiring lock again)
    if (!current_global_session_.empty()) {
        auto it = sessions_.find(current_global_session_);
        if (it != sessions_.end()) {
            it->second->Stop();
        }
        current_global_session_.clear();
    }
    
    // Create new session
    auto session = std::make_shared<ProfilerSession>(session_name);
    Status status = session->Start();
    if (status != Status::SUCCESS) {
        return status;
    }
    
    sessions_[session_name] = session;
    current_global_session_ = session_name;
    
    return Status::SUCCESS;
}

Status Profiler::StopGlobalSession() {
    if (!initialized_ || !enabled_) {
        return Status::NOT_INITIALIZED;
    }
    
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    if (current_global_session_.empty()) {
        return Status::FAILURE;
    }
    
    auto it = sessions_.find(current_global_session_);
    if (it != sessions_.end()) {
        it->second->Stop();
    }
    
    current_global_session_.clear();
    return Status::SUCCESS;
}

Status Profiler::StartSessionForRequest(uint64_t request_id) {
    if (!initialized_ || !enabled_) {
        return Status::NOT_INITIALIZED;
    }
    
    if (request_id == 0) {
        return Status::INVALID_ARGUMENT;
    }
    
    std::string session_name = "request_" + std::to_string(request_id);
    
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    // Create session for this request
    auto session = std::make_shared<ProfilerSession>(session_name);
    Status status = session->Start();
    if (status != Status::SUCCESS) {
        return status;
    }
    
    sessions_[session_name] = session;
    
    // Mark session start event
    MarkEvent(request_id, "session_start");
    
    return Status::SUCCESS;
}

Status Profiler::EndSessionForRequest(uint64_t request_id) {
    if (!initialized_ || !enabled_) {
        return Status::NOT_INITIALIZED;
    }
    
    if (request_id == 0) {
        return Status::INVALID_ARGUMENT;
    }
    
    std::string session_name = "request_" + std::to_string(request_id);
    
    // Mark session end event
    MarkEvent(request_id, "session_end");
    
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    auto it = sessions_.find(session_name);
    if (it != sessions_.end()) {
        it->second->Stop();
    }
    
    return Status::SUCCESS;
}

Status Profiler::MarkEvent(uint64_t request_id, const std::string& event_name) {
    if (!initialized_ || !enabled_) {
        return Status::NOT_INITIALIZED;
    }
    
    if (event_name.empty()) {
        return Status::INVALID_ARGUMENT;
    }
    
    ProfilerEvent event(GetCurrentTimestamp(), std::this_thread::get_id(), 
                       request_id, EventType::MARK, event_name);
    
    AddEventInternal(event);
    return Status::SUCCESS;
}

std::unique_ptr<ScopedEvent> Profiler::CreateScopedEvent(uint64_t request_id, 
                                                        const std::string& event_name) {
    if (!initialized_ || !enabled_) {
        return nullptr;
    }
    
    return std::make_unique<ScopedEvent>(request_id, event_name);
}

Status Profiler::ExportSessionAsJson(const std::string& session_name, 
                                    const std::string& file_path) {
    if (!initialized_) {
        return Status::NOT_INITIALIZED;
    }
    
    // Flush all thread buffers before exporting
    FlushAllThreadBuffers();
    
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    auto it = sessions_.find(session_name);
    if (it == sessions_.end()) {
        return Status::FAILURE;
    }
    
    return it->second->ExportToJson(file_path);
}

SessionMetadata Profiler::GetSessionMetadata(const std::string& session_name) const {
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    auto it = sessions_.find(session_name);
    if (it != sessions_.end()) {
        return it->second->GetMetadata();
    }
    
    return SessionMetadata{};
}

ThreadLocalBuffer* Profiler::GetThreadLocalBuffer() {
    if (!thread_buffer_) {
        thread_buffer_ = new ThreadLocalBuffer();
    }
    return thread_buffer_;
}

void Profiler::FlushAllThreadBuffers() {
    // Note: In a real implementation, this would need to coordinate with all threads
    // For now, we'll flush the current thread's buffer
    if (thread_buffer_) {
        auto events = thread_buffer_->ExtractEvents();
        if (!events.empty()) {
            // Add events to current global session if active
            std::lock_guard<std::mutex> lock(global_mutex_);
            if (!current_global_session_.empty()) {
                auto it = sessions_.find(current_global_session_);
                if (it != sessions_.end()) {
                    it->second->AddEvents(events);
                }
            }
        }
    }
}

uint64_t Profiler::GetCurrentTimestamp() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

void Profiler::AddEventInternal(const ProfilerEvent& event) {
    ThreadLocalBuffer* buffer = GetThreadLocalBuffer();
    if (!buffer->AddEvent(event)) {
        // Buffer overflow - mark in current session
        std::lock_guard<std::mutex> lock(global_mutex_);
        if (!current_global_session_.empty()) {
            auto it = sessions_.find(current_global_session_);
            if (it != sessions_.end()) {
                // Mark overflow in session metadata
                it->second->SetMetadata("overflow_detected", "true");
            }
        }
    }
}

// ScopedEvent implementation
ScopedEvent::ScopedEvent(uint64_t request_id, const std::string& event_name)
    : request_id_(request_id), event_name_(event_name), active_(true) {
    
    start_time_ns_ = Profiler::GetInstance().GetCurrentTimestamp();
    
    // Record start event
    ProfilerEvent start_event(start_time_ns_, std::this_thread::get_id(), 
                             request_id_, EventType::SCOPED_START, event_name_);
    Profiler::GetInstance().AddEventInternal(start_event);
}

ScopedEvent::~ScopedEvent() {
    if (active_) {
        uint64_t end_time_ns = Profiler::GetInstance().GetCurrentTimestamp();
        uint64_t duration_ns = end_time_ns - start_time_ns_;
        
        // Record end event
        ProfilerEvent end_event(end_time_ns, std::this_thread::get_id(), 
                               request_id_, EventType::SCOPED_END, event_name_, duration_ns);
        Profiler::GetInstance().AddEventInternal(end_event);
    }
}

ScopedEvent::ScopedEvent(ScopedEvent&& other) noexcept
    : request_id_(other.request_id_), event_name_(std::move(other.event_name_)),
      start_time_ns_(other.start_time_ns_), active_(other.active_) {
    other.active_ = false;
}

ScopedEvent& ScopedEvent::operator=(ScopedEvent&& other) noexcept {
    if (this != &other) {
        // End current event if active
        if (active_) {
            uint64_t end_time_ns = Profiler::GetInstance().GetCurrentTimestamp();
            uint64_t duration_ns = end_time_ns - start_time_ns_;
            
            ProfilerEvent end_event(end_time_ns, std::this_thread::get_id(), 
                                   request_id_, EventType::SCOPED_END, event_name_, duration_ns);
            Profiler::GetInstance().AddEventInternal(end_event);
        }
        
        request_id_ = other.request_id_;
        event_name_ = std::move(other.event_name_);
        start_time_ns_ = other.start_time_ns_;
        active_ = other.active_;
        
        other.active_ = false;
    }
    return *this;
}

} // namespace edge_ai