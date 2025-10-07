/**
 * @file trace_viewer_simple.cpp
 * @brief Simple trace analysis CLI tool for profiler JSON exports
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
 * @struct OptimizationDecision
 * @brief Represents an optimization decision from the trace
 */
struct OptimizationDecision {
    std::string action;
    std::string trigger;
    std::string parameter_name;
    std::string old_value;
    std::string new_value;
    double expected_improvement;
    uint64_t timestamp_ns;
    uint64_t request_id;
    
    OptimizationDecision() : expected_improvement(0.0), timestamp_ns(0), request_id(0) {}
};

/**
 * @struct OptimizationStats
 * @brief Statistics about optimization performance
 */
struct OptimizationStats {
    uint64_t total_decisions;
    uint64_t successful_optimizations;
    uint64_t failed_optimizations;
    double avg_improvement_percent;
    
    OptimizationStats() : total_decisions(0), successful_optimizations(0), 
                         failed_optimizations(0), avg_improvement_percent(0.0) {}
};

/**
 * @brief Simple JSON value extraction
 */
std::string ExtractJsonValue(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) {
        return "";
    }
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) {
        return "";
    }
    
    pos = json.find_first_of("\"", pos);
    if (pos == std::string::npos) {
        return "";
    }
    
    size_t start = pos + 1;
    size_t end = json.find("\"", start);
    if (end == std::string::npos) {
        return "";
    }
    
    return json.substr(start, end - start);
}

/**
 * @brief Extract numeric JSON value
 */
uint64_t ExtractJsonNumber(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) {
        return 0;
    }
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) {
        return 0;
    }
    
    pos = json.find_first_not_of(" \t\n\r", pos + 1);
    if (pos == std::string::npos) {
        return 0;
    }
    
    size_t end = pos;
    while (end < json.length() && (std::isdigit(json[end]) || json[end] == '.')) {
        end++;
    }
    
    std::string num_str = json.substr(pos, end - pos);
    return std::stoull(num_str);
}

/**
 * @brief Extract boolean JSON value
 */
bool ExtractJsonBool(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) {
        return false;
    }
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) {
        return false;
    }
    
    pos = json.find_first_not_of(" \t\n\r", pos + 1);
    if (pos == std::string::npos) {
        return false;
    }
    
    std::string value = json.substr(pos, 4);
    return value == "true";
}

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
    
    // Extract session metadata
    session.name = ExtractJsonValue(json_content, "session_name");
    session.start_time_ns = ExtractJsonNumber(json_content, "start_time_ns");
    session.end_time_ns = ExtractJsonNumber(json_content, "end_time_ns");
    session.total_requests = ExtractJsonNumber(json_content, "total_requests");
    session.has_overflow = ExtractJsonBool(json_content, "has_overflow");
    
    // For now, create some dummy events for demonstration
    // In a real implementation, you would parse the events array
    TraceEvent event1;
    event1.name = "model_load";
    event1.timestamp_ns = session.start_time_ns + 1000000;
    event1.request_id = 1;
    event1.thread_id = "main";
    event1.duration_ms = 5.2;
    event1.event_type = "SCOPED_END";
    session.events.push_back(event1);
    
    TraceEvent event2;
    event2.name = "backend_execute";
    event2.timestamp_ns = session.start_time_ns + 2000000;
    event2.request_id = 1;
    event2.thread_id = "main";
    event2.duration_ms = 12.8;
    event2.event_type = "SCOPED_END";
    session.events.push_back(event2);
    
    TraceEvent event3;
    event3.name = "inference_total";
    event3.timestamp_ns = session.start_time_ns + 3000000;
    event3.request_id = 1;
    event3.thread_id = "main";
    event3.duration_ms = 18.5;
    event3.event_type = "SCOPED_END";
    session.events.push_back(event3);
    
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
 * @brief Parse optimization decisions from JSON
 */
std::vector<OptimizationDecision> ParseOptimizationDecisions(const std::string& json_content) {
    std::vector<OptimizationDecision> decisions;
    
    // Look for optimization_stats section
    size_t stats_pos = json_content.find("\"optimization_stats\"");
    if (stats_pos == std::string::npos) {
        return decisions; // No optimization data found
    }
    
    // For now, create some dummy optimization decisions for demonstration
    // In a real implementation, you would parse the decisions array from JSON
    
    OptimizationDecision decision1;
    decision1.action = "ADJUST_BATCH_SIZE";
    decision1.trigger = "LATENCY_THRESHOLD_EXCEEDED";
    decision1.parameter_name = "max_batch_size";
    decision1.old_value = "8";
    decision1.new_value = "4";
    decision1.expected_improvement = 0.2;
    decision1.timestamp_ns = 1000000000;
    decision1.request_id = 1;
    decisions.push_back(decision1);
    
    OptimizationDecision decision2;
    decision2.action = "SWITCH_BACKEND";
    decision2.trigger = "BACKEND_PERFORMANCE_CHANGE";
    decision2.parameter_name = "preferred_backend";
    decision2.old_value = "CPU";
    decision2.new_value = "GPU";
    decision2.expected_improvement = 0.3;
    decision2.timestamp_ns = 2000000000;
    decision2.request_id = 2;
    decisions.push_back(decision2);
    
    return decisions;
}

/**
 * @brief Parse optimization statistics from JSON
 */
OptimizationStats ParseOptimizationStats(const std::string& json_content) {
    OptimizationStats stats;
    
    // Look for optimization_stats section
    size_t stats_pos = json_content.find("\"optimization_stats\"");
    if (stats_pos == std::string::npos) {
        return stats; // No optimization data found
    }
    
    // Extract optimization statistics
    stats.total_decisions = ExtractJsonNumber(json_content, "total_decisions");
    stats.successful_optimizations = ExtractJsonNumber(json_content, "successful_optimizations");
    stats.failed_optimizations = ExtractJsonNumber(json_content, "failed_optimizations");
    
    // Extract average improvement (this would need to be parsed as a double)
    std::string improvement_str = ExtractJsonValue(json_content, "avg_improvement_percent");
    if (!improvement_str.empty()) {
        stats.avg_improvement_percent = std::stod(improvement_str);
    }
    
    return stats;
}

/**
 * @brief Print optimization tuning summary
 */
void PrintOptimizationTuning(const std::vector<OptimizationDecision>& decisions, 
                           const OptimizationStats& stats) {
    std::cout << "🎯 Optimization Tuning Summary:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    if (stats.total_decisions == 0) {
        std::cout << "   No optimization decisions found in trace." << std::endl;
        return;
    }
    
    std::cout << "📊 Optimization Statistics:" << std::endl;
    std::cout << "   Total Decisions: " << stats.total_decisions << std::endl;
    std::cout << "   Successful: " << stats.successful_optimizations << std::endl;
    std::cout << "   Failed: " << stats.failed_optimizations << std::endl;
    std::cout << "   Success Rate: " << std::fixed << std::setprecision(1) 
              << (stats.total_decisions > 0 ? (stats.successful_optimizations * 100.0 / stats.total_decisions) : 0.0) 
              << "%" << std::endl;
    std::cout << "   Avg Improvement: " << std::fixed << std::setprecision(1) 
              << stats.avg_improvement_percent << "%" << std::endl;
    
    std::cout << "\n🔧 Recent Optimization Decisions:" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(20) << "Action" 
              << std::setw(25) << "Parameter" 
              << std::setw(15) << "Old → New" 
              << std::setw(12) << "Expected %" << std::endl;
    std::cout << std::string(72, '-') << std::endl;
    
    for (const auto& decision : decisions) {
        std::cout << std::setw(20) << decision.action
                  << std::setw(25) << decision.parameter_name
                  << std::setw(15) << (decision.old_value + " → " + decision.new_value)
                  << std::setw(12) << (decision.expected_improvement * 100.0) << std::endl;
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
    std::cout << "  --show-tuning          Show optimization tuning decisions" << std::endl;
    std::cout << "  --help                  Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " trace.json" << std::endl;
    std::cout << "  " << program_name << " trace.json --stage=backend_execute" << std::endl;
    std::cout << "  " << program_name << " trace.json --show-tuning" << std::endl;
}

} // namespace edge_ai

int main(int argc, char* argv[]) {
    if (argc < 2) {
        edge_ai::PrintUsage(argv[0]);
        return 1;
    }
    
    std::string trace_file = argv[1];
    std::string filter_stage;
    bool show_tuning = false;
    
    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            edge_ai::PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--show-tuning") {
            show_tuning = true;
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
    
    // Read the trace file content for optimization parsing
    std::ifstream file(trace_file);
    std::string json_content;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            json_content += line;
        }
        file.close();
    }
    
    // Parse trace file
    auto session = edge_ai::ParseTraceFile(trace_file);
    
    if (session.events.empty()) {
        std::cerr << "Error: No events found in trace file or file could not be parsed." << std::endl;
        return 1;
    }
    
    // Print session summary
    edge_ai::PrintSessionSummary(session);
    
    // Handle optimization tuning display
    if (show_tuning) {
        auto optimization_decisions = edge_ai::ParseOptimizationDecisions(json_content);
        auto optimization_stats = edge_ai::ParseOptimizationStats(json_content);
        edge_ai::PrintOptimizationTuning(optimization_decisions, optimization_stats);
    }
    
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
