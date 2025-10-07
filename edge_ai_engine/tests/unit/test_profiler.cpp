/**
 * @file test_profiler.cpp
 * @brief Unit tests for the Profiler system
 * @author AI Co-Developer
 * @date 2024
 */

#include <gtest/gtest.h>
#include "profiling/profiler.h"
#include "profiling/profiler_session.h"
#include <thread>
#include <chrono>
#include <fstream>

namespace edge_ai {

class ProfilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize profiler for each test
        profiler_ = &Profiler::GetInstance();
        ASSERT_EQ(profiler_->Initialize(), Status::SUCCESS);
    }
    
    void TearDown() override {
        // Clean up profiler
        if (profiler_) {
            profiler_->Shutdown();
        }
    }
    
    Profiler* profiler_;
};

TEST_F(ProfilerTest, BasicInitialization) {
    EXPECT_TRUE(profiler_->IsEnabled());
    EXPECT_EQ(profiler_->Initialize(), Status::ALREADY_INITIALIZED);
}

TEST_F(ProfilerTest, GlobalSessionManagement) {
    // Start global session
    EXPECT_EQ(profiler_->StartGlobalSession("test_session"), Status::SUCCESS);
    
    // Try to start another global session (should stop current one)
    EXPECT_EQ(profiler_->StartGlobalSession("test_session2"), Status::SUCCESS);
    
    // Stop global session
    EXPECT_EQ(profiler_->StopGlobalSession(), Status::SUCCESS);
    
    // Try to stop when no session is active
    EXPECT_EQ(profiler_->StopGlobalSession(), Status::FAILURE);
}

TEST_F(ProfilerTest, RequestSessionManagement) {
    const uint64_t request_id = 12345;
    
    // Start session for request
    EXPECT_EQ(profiler_->StartSessionForRequest(request_id), Status::SUCCESS);
    
    // End session for request
    EXPECT_EQ(profiler_->EndSessionForRequest(request_id), Status::SUCCESS);
    
    // Try to end session for non-existent request
    EXPECT_EQ(profiler_->EndSessionForRequest(99999), Status::SUCCESS);
}

TEST_F(ProfilerTest, MarkEvents) {
    const uint64_t request_id = 54321;
    
    // Start global session
    EXPECT_EQ(profiler_->StartGlobalSession("mark_test"), Status::SUCCESS);
    
    // Mark events
    EXPECT_EQ(profiler_->MarkEvent(0, "global_event"), Status::SUCCESS);
    EXPECT_EQ(profiler_->MarkEvent(request_id, "request_event"), Status::SUCCESS);
    
    // Mark event with empty name (should fail)
    EXPECT_EQ(profiler_->MarkEvent(request_id, ""), Status::INVALID_ARGUMENT);
    
    // Stop session
    EXPECT_EQ(profiler_->StopGlobalSession(), Status::SUCCESS);
}

TEST_F(ProfilerTest, ScopedEvent) {
    const uint64_t request_id = 67890;
    
    // Start global session
    EXPECT_EQ(profiler_->StartGlobalSession("scoped_test"), Status::SUCCESS);
    
    // Create scoped event
    auto scoped_event = profiler_->CreateScopedEvent(request_id, "test_operation");
    EXPECT_NE(scoped_event, nullptr);
    
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Scoped event should automatically record end when destroyed
    scoped_event.reset();
    
    // Stop session
    EXPECT_EQ(profiler_->StopGlobalSession(), Status::SUCCESS);
}

TEST_F(ProfilerTest, EventSourceRegistration) {
    EXPECT_EQ(profiler_->RegisterEventSource("test_source"), Status::SUCCESS);
    EXPECT_EQ(profiler_->RegisterEventSource(""), Status::INVALID_ARGUMENT);
}

TEST_F(ProfilerTest, JSONExport) {
    const std::string session_name = "export_test";
    const std::string file_path = "/tmp/test_profiler_export.json";
    
    // Start session and add some events
    EXPECT_EQ(profiler_->StartGlobalSession(session_name), Status::SUCCESS);
    EXPECT_EQ(profiler_->MarkEvent(0, "test_event"), Status::SUCCESS);
    
    // Export to JSON
    EXPECT_EQ(profiler_->ExportSessionAsJson(session_name, file_path), Status::SUCCESS);
    
    // Verify file was created and contains expected content
    std::ifstream file(file_path);
    EXPECT_TRUE(file.is_open());
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("session_metadata") != std::string::npos);
    EXPECT_TRUE(content.find("events") != std::string::npos);
    EXPECT_TRUE(content.find("test_event") != std::string::npos);
    
    file.close();
    
    // Clean up
    std::remove(file_path.c_str());
    EXPECT_EQ(profiler_->StopGlobalSession(), Status::SUCCESS);
}

TEST_F(ProfilerTest, ThreadSafety) {
    const std::string session_name = "thread_test";
    const int num_threads = 4;
    const int events_per_thread = 10;
    
    // Start global session
    EXPECT_EQ(profiler_->StartGlobalSession(session_name), Status::SUCCESS);
    
    std::vector<std::thread> threads;
    
    // Create multiple threads that mark events
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, i, events_per_thread]() {
            for (int j = 0; j < events_per_thread; ++j) {
                uint64_t request_id = i * events_per_thread + j;
                profiler_->MarkEvent(request_id, "thread_event_" + std::to_string(i));
                
                // Create scoped event
                auto scoped = profiler_->CreateScopedEvent(request_id, "scoped_" + std::to_string(i));
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                scoped.reset();
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Export and verify
    const std::string file_path = "/tmp/thread_test_export.json";
    EXPECT_EQ(profiler_->ExportSessionAsJson(session_name, file_path), Status::SUCCESS);
    
    // Verify file was created
    std::ifstream file(file_path);
    EXPECT_TRUE(file.is_open());
    file.close();
    
    // Clean up
    std::remove(file_path.c_str());
    EXPECT_EQ(profiler_->StopGlobalSession(), Status::SUCCESS);
}

TEST_F(ProfilerTest, DisabledProfiler) {
    // Disable profiler
    profiler_->SetEnabled(false);
    EXPECT_FALSE(profiler_->IsEnabled());
    
    // Operations should still work but not record events
    EXPECT_EQ(profiler_->StartGlobalSession("disabled_test"), Status::NOT_INITIALIZED);
    EXPECT_EQ(profiler_->MarkEvent(0, "disabled_event"), Status::NOT_INITIALIZED);
    
    // Re-enable profiler
    profiler_->SetEnabled(true);
    EXPECT_TRUE(profiler_->IsEnabled());
    
    // Operations should work again
    EXPECT_EQ(profiler_->StartGlobalSession("enabled_test"), Status::SUCCESS);
    EXPECT_EQ(profiler_->MarkEvent(0, "enabled_event"), Status::SUCCESS);
    EXPECT_EQ(profiler_->StopGlobalSession(), Status::SUCCESS);
}

} // namespace edge_ai
