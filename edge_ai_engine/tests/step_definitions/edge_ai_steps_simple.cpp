#include <testing/behavior_driven_testing.h>
#include <profiling/profiler.h>
#include <iostream>
#include <memory>

using namespace edge_ai;
using namespace edge_ai::testing;

// Global test context
static std::map<std::string, std::string> g_test_context;

// Helper function to get test context
std::map<std::string, std::string>& GetTestContext() {
    return g_test_context;
}

// Simple step registration function
void RegisterEdgeAISteps() {
    BDTManager* manager = GetBDTManager();
    if (!manager) {
        std::cerr << "Failed to get BDT Manager" << std::endl;
        return;
    }
    
    BDTStepRegistry* registry = manager->GetStepRegistry();
    if (!registry) {
        std::cerr << "Failed to get Step Registry" << std::endl;
        return;
    }
    
    // Register Given steps
    registry->RegisterGivenStep(
        R"(the Edge AI engine is initialized)",
        []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status {
            PROFILER_SCOPED_EVENT(0, "bdt_given_engine_initialized");
            
            try {
                // TODO: Initialize Edge AI engine properly
                GetTestContext()["engine_initialized"] = "true";
                return Status::SUCCESS;
            } catch (const std::exception& e) {
                std::cerr << "Exception during engine initialization: " << e.what() << std::endl;
                return Status::FAILURE;
            }
        }
    );
    
    registry->RegisterGivenStep(
        R"(the system is running)",
        []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status {
            PROFILER_SCOPED_EVENT(0, "bdt_given_system_running");
            
            if (GetTestContext()["engine_initialized"] != "true") {
                std::cerr << "Engine not initialized" << std::endl;
                return Status::FAILURE;
            }
            
            GetTestContext()["system_running"] = "true";
            return Status::SUCCESS;
        }
    );
    
    registry->RegisterGivenStep(
        R"(an Edge AI model is loaded)",
        []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status {
            PROFILER_SCOPED_EVENT(0, "bdt_given_model_loaded");
            
            try {
                // TODO: Load Edge AI model properly
                GetTestContext()["model_loaded"] = "true";
                GetTestContext()["model_path"] = "test_models/sample_model.onnx";
                return Status::SUCCESS;
            } catch (const std::exception& e) {
                std::cerr << "Exception during model loading: " << e.what() << std::endl;
                return Status::FAILURE;
            }
        }
    );
    
    registry->RegisterGivenStep(
        R"(the model is ready for inference)",
        []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status {
            PROFILER_SCOPED_EVENT(0, "bdt_given_model_ready");
            
            if (GetTestContext()["model_loaded"] != "true") {
                std::cerr << "Model not loaded" << std::endl;
                return Status::FAILURE;
            }
            
            GetTestContext()["model_ready"] = "true";
            return Status::SUCCESS;
        }
    );
    
    // Register When steps
    registry->RegisterWhenStep(
        R"(I run inference on the model)",
        []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status {
            PROFILER_SCOPED_EVENT(0, "bdt_when_run_inference");
            
            try {
                if (GetTestContext()["model_ready"] != "true") {
                    std::cerr << "Model not ready for inference" << std::endl;
                    return Status::FAILURE;
                }
                
                // TODO: Run actual inference
                GetTestContext()["inference_time_ms"] = "50";
                GetTestContext()["inference_completed"] = "true";
                
                return Status::SUCCESS;
            } catch (const std::exception& e) {
                std::cerr << "Exception during inference: " << e.what() << std::endl;
                return Status::FAILURE;
            }
        }
    );
    
    // Register Then steps
    registry->RegisterThenStep(
        R"(the inference should complete successfully)",
        []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status {
            PROFILER_SCOPED_EVENT(0, "bdt_then_inference_success");
            
            if (GetTestContext()["inference_completed"] != "true") {
                std::cerr << "No inference completed" << std::endl;
                return Status::FAILURE;
            }
            
            return Status::SUCCESS;
        }
    );
    
    registry->RegisterThenStep(
        R"(the result should be valid)",
        []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status {
            PROFILER_SCOPED_EVENT(0, "bdt_then_result_valid");
            
            // TODO: Validate inference results
            return Status::SUCCESS;
        }
    );
    
    registry->RegisterThenStep(
        R"(the inference should complete within (\d+) milliseconds)",
        [](const std::map<std::string, std::string>& params) -> Status {
            PROFILER_SCOPED_EVENT(0, "bdt_then_inference_timing");
            
            auto it = params.find("time_limit");
            if (it == params.end()) {
                std::cerr << "Time limit parameter not found" << std::endl;
                return Status::FAILURE;
            }
            
            uint32_t time_limit = std::stoul(it->second);
            auto time_it = GetTestContext().find("inference_time_ms");
            if (time_it != GetTestContext().end()) {
                uint32_t actual_time = std::stoul(time_it->second);
                if (actual_time > time_limit) {
                    std::cerr << "Inference time " << actual_time << "ms exceeds limit " << time_limit << "ms" << std::endl;
                    return Status::FAILURE;
                }
            }
            
            return Status::SUCCESS;
        }
    );
}

// Auto-register steps when this file is loaded
static bool steps_registered = []() {
    RegisterEdgeAISteps();
    return true;
}();
