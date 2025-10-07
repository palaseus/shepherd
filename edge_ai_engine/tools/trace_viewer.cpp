/**
 * @file trace_viewer.cpp
 * @brief Minimal trace analysis CLI tool for profiler JSON exports
 * @author AI Co-Developer
 * @date 2024
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <regex>

namespace edge_ai {

/**
 * @struct TraceEvent
 * @brief Represents a single profiler event
 */
struct TraceEvent {
    std::string name;
    uint64_t timestamp_ns;
    uint64_t request_id;
    std::string thread_id;
    double duration_ms;
    std::string event_type;
    
    TraceEvent() : timestamp_ns(0), request_id(0), duration_ms(0.0) {}
};

/**
 * @struct TraceSession
 * @brief Represents a profiler session with events
 */
struct TraceSession {
    std::string name;
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    uint64_t total_requests;
    bool has_overflow;
    std::vector<TraceEvent> events;
    
    TraceSession() : start_time_ns(0), end_time_ns(0), total_requests(0), has_overflow(false) {}
};

/**
 * @struct StageStats
 * @brief Statistics for a specific stage
 */
struct StageStats {
    std::string stage_name;
    int count;
    double total_time_ms;
    double min_time_ms;
    double max_time_ms;
    double avg_time_ms;
    double p50_time_ms;
    double p90_time_ms;
    double p99_time_ms;
    
    StageStats() : count(0), total_time_ms(0.0), min_time_ms(0.0), max_time_ms(0.0),
                   avg_time_ms(0.0), p50_time_ms(0.0), p90_time_ms(0.0), p99_time_ms(0.0) {}
};

/**
 * @brief Parse JSON file and extract trace session
 */
TraceSession ParseTraceFile(const std::string& file_path) {
    TraceSession session;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return session;
    }
    
    std::string line;
    std::string json_content;
    
    // Read entire file
    while (std::getline(file, line)) {
        json_content += line;
    }
    file.close();
    
    // Simple JSON parsing (basic implementation)
    // Extract session metadata
    std::regex session_name_regex(R"("session_name"\s*:\s*"([^"]+)")");
    std::smatch match;
    if (std::regex_search(json_content, match, session_name_regex)) {
        session.name = match[1].str();
    }
    
    std::regex start_time_regex(R"("start_time_ns"\s*:\s*(\d+))");
    if (std::regex_search(json_content, match, start_time_regex)) {
        session.start_time_ns = std::stoull(match[1].str());
    }
    
    std::regex end_time_regex(R"("end_time_ns"\s*:\s*(\d+))");
    if (std::regex_search(json_content, match, end_time_regex)) {
        session.end_time_ns = std::stoull(match[1].str());
    }
    
    std::regex total_requests_regex(R"("total_requests"\s*:\s*(\d+))");
    if (std::regex_search(json_content, match, total_requests_regex)) {
        session.total_requests = std::stoull(match[1].str());
    }
    
    std::regex overflow_regex(R"("has_overflow"\s*:\s*(true|false))");
    if (std::regex_search(json_content, match, overflow_regex)) {
        session.has_overflow = (match[1].str() == "true");
    }
    
    // Extract events (simplified parsing)
    std::regex event_regex(R"(\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"timestamp_ns"\s*:\s*(\d+)\s*,\s*"request_id"\s*:\s*(\d+)\s*,\s*"thread_id"\s*:\s*"([^"]+)"\s*,\s*"duration_ms"\s*:\s*([\d.]+)\s*,\s*"event_type"\s*:\s*"([^"]+)"\s*\})");
    std::sregex_iterator iter(json_content.begin(), json_content.end(), event_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        TraceEvent event;
        event.name = (*iter)[1].str();
        event.timestamp_ns = std::stoull((*iter)[2].str());
        event.request_id = std::stoull((*iter)[3].str());
        event.thread_id = (*iter)[4].str();
        event.duration_ms = std::stod((*iter)[5].str());
        event.event_type = (*iter)[6].str();
        
        session.events.push_back(event);
    }
    
    return session;
}

/**
 * @brief Calculate statistics for each stage
 */
std::map<std::string, StageStats> CalculateStageStats(const TraceSession& session) {
    std::map<std::string, StageStats> stage_stats;
    
    for (const auto& event : session.events) {
        if (event.event_type == "SCOPED_END" && event.duration_ms > 0.0) {
            auto& stats = stage_stats[event.name];
            stats.stage_name = event.name;
            stats.count++;
            stats.total_time_ms += event.duration_ms;
            
            if (stats.count == 1) {
                stats.min_time_ms = event.duration_ms;
                stats.max_time_ms = event.duration_ms;
            } else {
                stats.min_time_ms = std::min(stats.min_time_ms, event.duration_ms);
                stats.max_time_ms = std::max(stats.max_time_ms, event.duration_ms);
            }
        }
    }
    
    // Calculate averages and percentiles
    for (auto& [stage_name, stats] : stage_stats) {
        if (stats.count > 0) {
            stats.avg_time_ms = stats.total_time_ms / stats.count;
            
            // Collect durations for percentile calculation
            std::vector<double> durations;
            for (const auto& event : session.events) {
                if (event.name == stage_name && event.event_type == "SCOPED_END" && event.duration_ms > 0.0) {
                    durations.push_back(event.duration_ms);
                }
            }
            
            if (!durations.empty()) {
                std::sort(durations.begin(), durations.end());
                size_t n = durations.size();
                stats.p50_time_ms = durations[static_cast<size_t>(n * 0.50)];
                stats.p90_time_ms = durations[static_cast<size_t>(n * 0.90)];
                stats.p99_time_ms = durations[static_cast<size_t>(n * 0.99)];
            }
        }
    }
    
    return stage_stats;
}

/**
 * @brief Print session summary
 */
void PrintSessionSummary(const TraceSession& session) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                         TRACE SESSION SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "\n📊 Session Information:" << std::endl;
    std::cout << "   Name: " << session.name << std::endl;
    std::cout << "   Total Requests: " << session.total_requests << std::endl;
    std::cout << "   Total Events: " << session.events.size() << std::endl;
    
    if (session.start_time_ns > 0 && session.end_time_ns > 0) {
        double duration_ms = (session.end_time_ns - session.start_time_ns) / 1000000.0;
        std::cout << "   Duration: " << std::fixed << std::setprecision(2) << duration_ms << " ms" << std::endl;
    }
    
    if (session.has_overflow) {
        std::cout << "   ⚠️  WARNING: Buffer overflow detected!" << std::endl;
    }
    
    std::cout << std::endl;
}

/**
 * @brief Print top slowest stages
 */
void PrintTopSlowestStages(const std::map<std::string, StageStats>& stage_stats, int top_n = 5) {
    std::cout << "🐌 Top " << top_n << " Slowest Stages (by average time):" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // Sort stages by average time
    std::vector<std::pair<std::string, StageStats>> sorted_stages;
    for (const auto& [stage_name, stats] : stage_stats) {
        if (stats.count > 0) {
            sorted_stages.push_back({stage_name, stats});
        }
    }
    
    std::sort(sorted_stages.begin(), sorted_stages.end(),
              [](const auto& a, const auto& b) {
                  return a.second.avg_time_ms > b.second.avg_time_ms;
              });
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(25) << "Stage" 
              << std::setw(10) << "Count" 
              << std::setw(12) << "Avg (ms)" 
              << std::setw(12) << "P99 (ms)" 
              << std::setw(12) << "Total (ms)" << std::endl;
    std::cout << std::string(71, '-') << std::endl;
    
    int count = 0;
    for (const auto& [stage_name, stats] : sorted_stages) {
        if (count >= top_n) break;
        
        std::cout << std::setw(25) << stage_name
                  << std::setw(10) << stats.count
                  << std::setw(12) << stats.avg_time_ms
                  << std::setw(12) << stats.p99_time_ms
                  << std::setw(12) << stats.total_time_ms << std::endl;
        count++;
    }
    
    std::cout << std::endl;
}

/**
 * @brief Print backend execution summary
 */
void PrintBackendSummary(const std::map<std::string, StageStats>& stage_stats) {
    std::cout << "⚡ Backend Execution Summary:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // Look for backend execution events
    std::vector<std::pair<std::string, StageStats>> backend_stages;
    for (const auto& [stage_name, stats] : stage_stats) {
        if (stage_name.find("backend") != std::string::npos || 
            stage_name.find("execute") != std::string::npos) {
            backend_stages.push_back({stage_name, stats});
        }
    }
    
    if (backend_stages.empty()) {
        std::cout << "   No backend execution events found." << std::endl;
        return;
    }
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(25) << "Backend Stage" 
              << std::setw(10) << "Count" 
              << std::setw(12) << "Avg (ms)" 
              << std::setw(12) << "Total (ms)" << std::endl;
    std::cout << std::string(59, '-') << std::endl;
    
    for (const auto& [stage_name, stats] : backend_stages) {
        std::cout << std::setw(25) << stage_name
                  << std::setw(10) << stats.count
                  << std::setw(12) << stats.avg_time_ms
                  << std::setw(12) << stats.total_time_ms << std::endl;
    }
    
    std::cout << std::endl;
}

/**
 * @brief Print usage information
 */
void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <trace.json> [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --stage=<stage_name>    Filter events for specific stage" << std::endl;
    std::cout << "  --help                  Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " trace.json" << std::endl;
    std::cout << "  " << program_name << " trace.json --stage=backend_execute" << std::endl;
}

} // namespace edge_ai

int main(int argc, char* argv[]) {
    if (argc < 2) {
        edge_ai::PrintUsage(argv[0]);
        return 1;
    }
    
    std::string trace_file = argv[1];
    std::string filter_stage;
    
    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            edge_ai::PrintUsage(argv[0]);
            return 0;
        } else if (arg.substr(0, 8) == "--stage=") {
            filter_stage = arg.substr(8);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            edge_ai::PrintUsage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "🔍 Edge AI Engine Trace Viewer" << std::endl;
    std::cout << "📁 Analyzing trace file: " << trace_file << std::endl;
    
    // Parse trace file
    auto session = edge_ai::ParseTraceFile(trace_file);
    
    if (session.events.empty()) {
        std::cerr << "Error: No events found in trace file or file could not be parsed." << std::endl;
        return 1;
    }
    
    // Print session summary
    edge_ai::PrintSessionSummary(session);
    
    // Calculate stage statistics
    auto stage_stats = edge_ai::CalculateStageStats(session);
    
    if (stage_stats.empty()) {
        std::cout << "No stage statistics available (no SCOPED_END events found)." << std::endl;
        return 0;
    }
    
    // Apply stage filter if specified
    if (!filter_stage.empty()) {
        std::cout << "🔍 Filtering for stage: " << filter_stage << std::endl;
        
        auto it = stage_stats.find(filter_stage);
        if (it != stage_stats.end()) {
            std::map<std::string, edge_ai::StageStats> filtered_stats;
            filtered_stats[filter_stage] = it->second;
            stage_stats = filtered_stats;
        } else {
            std::cout << "⚠️  Stage '" << filter_stage << "' not found in trace." << std::endl;
            return 1;
        }
    }
    
    // Print analysis results
    edge_ai::PrintTopSlowestStages(stage_stats);
    edge_ai::PrintBackendSummary(stage_stats);
    
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "✅ Trace analysis complete!" << std::endl;
    
    return 0;
}
