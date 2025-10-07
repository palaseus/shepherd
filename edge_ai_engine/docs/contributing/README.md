# Contributing Guide

This document provides guidelines and instructions for contributing to the Edge AI Inference Engine project.

## Table of Contents

- [Getting Started](getting_started.md)
- [Development Setup](development_setup.md)
- [Code Style](code_style.md)
- [Testing Guidelines](testing_guidelines.md)
- [Pull Request Process](pull_request_process.md)
- [Issue Reporting](issue_reporting.md)
- [Documentation](documentation.md)

## Getting Started

### Prerequisites

Before contributing to the Edge AI Inference Engine, ensure you have:

- **C++20 compatible compiler** (GCC 10+ or Clang 12+)
- **CMake 3.20+**
- **Python 3.8+** (for utilities and testing)
- **Git** for version control
- **Basic understanding** of C++ and machine learning concepts

### Development Environment

#### Linux (Ubuntu 20.04+)
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    libomp-dev \
    libcuda-dev

# Install optional dependencies
sudo apt-get install -y \
    cuda-toolkit \
    libopencl-dev \
    vulkan-utils
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install dependencies via Homebrew
brew install cmake python3 git

# Install optional dependencies
brew install openmp cuda
```

#### Windows
```bash
# Install Visual Studio 2019+ with C++ support
# Install CMake
# Install Python 3.8+
# Install Git for Windows
```

## Development Setup

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/edge-ai-engine.git
   cd edge-ai-engine
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-repo/edge-ai-engine.git
   ```

### Build Setup

1. **Create build directory**:
   ```bash
   mkdir build && cd build
   ```

2. **Configure build**:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Debug \
         -DPROFILER_ENABLED=ON \
         -DGPU_SUPPORT=ON \
         ..
   ```

3. **Build the project**:
   ```bash
   make -j$(nproc)
   ```

4. **Run tests**:
   ```bash
   make test
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a pull request** on GitHub

## Code Style

### C++ Code Style

The project follows a consistent C++ coding style based on modern C++ best practices.

#### Naming Conventions

- **Classes**: PascalCase (e.g., `EdgeAIEngine`, `OptimizationManager`)
- **Functions**: camelCase (e.g., `runInference`, `loadModel`)
- **Variables**: camelCase (e.g., `inputTensor`, `outputData`)
- **Constants**: UPPER_CASE (e.g., `MAX_BATCH_SIZE`, `DEFAULT_TIMEOUT`)
- **Namespaces**: snake_case (e.g., `edge_ai`, `edge_ai::testing`)

#### File Organization

```
src/
├── core/
│   ├── edge_ai_engine.cpp
│   ├── edge_ai_engine.h
│   ├── model_loader.cpp
│   └── model_loader.h
├── optimization/
│   ├── optimization_manager.cpp
│   └── optimization_manager.h
└── ...
```

#### Header Guards

Use `#pragma once` for header guards:

```cpp
#pragma once

#include <vector>
#include <string>

namespace edge_ai {

class MyClass {
public:
    // Class implementation
};

} // namespace edge_ai
```

#### Include Order

1. **System headers** (e.g., `<iostream>`, `<vector>`)
2. **Third-party headers** (e.g., `<gtest/gtest.h>`)
3. **Project headers** (e.g., `"core/edge_ai_engine.h"`)

```cpp
#include <iostream>
#include <vector>
#include <memory>

#include <gtest/gtest.h>

#include "core/edge_ai_engine.h"
#include "optimization/optimization_manager.h"
```

#### Function Documentation

Use Doxygen-style comments for public APIs:

```cpp
/**
 * @brief Runs inference on the loaded model
 * @param inputs Vector of input tensors
 * @param outputs Vector of output tensors (will be populated)
 * @return Status indicating success or failure
 * @throws std::invalid_argument if inputs are invalid
 */
Status RunInference(const std::vector<Tensor>& inputs, 
                   std::vector<Tensor>& outputs);
```

#### Error Handling

Use the `Status` enum for error handling:

```cpp
Status LoadModel(const std::string& model_path, ModelType type) {
    if (model_path.empty()) {
        return Status::ERROR_INVALID_ARGUMENT;
    }
    
    if (!std::filesystem::exists(model_path)) {
        return Status::ERROR_MODEL_NOT_FOUND;
    }
    
    // Load model logic
    return Status::OK;
}
```

#### Memory Management

- Use smart pointers (`std::unique_ptr`, `std::shared_ptr`)
- Follow RAII principles
- Avoid raw pointers in public APIs

```cpp
class MyClass {
private:
    std::unique_ptr<Implementation> impl_;
    
public:
    MyClass() : impl_(std::make_unique<Implementation>()) {}
    
    // Use move semantics
    MyClass(MyClass&&) = default;
    MyClass& operator=(MyClass&&) = default;
    
    // Disable copy
    MyClass(const MyClass&) = delete;
    MyClass& operator=(const MyClass&) = delete;
};
```

### Python Code Style

Follow PEP 8 for Python code:

```python
def load_model(model_path: str, model_type: ModelType) -> Status:
    """
    Load a model from the specified path.
    
    Args:
        model_path: Path to the model file
        model_type: Type of the model (ONNX, TensorFlow Lite, etc.)
        
    Returns:
        Status indicating success or failure
        
    Raises:
        ValueError: If model_path is empty
        FileNotFoundError: If model file doesn't exist
    """
    if not model_path:
        raise ValueError("Model path cannot be empty")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model logic
    return Status.OK
```

### Code Formatting

#### C++ Formatting

Use `clang-format` with the project's configuration:

```bash
# Format C++ code
clang-format -i src/**/*.cpp include/**/*.h

# Check formatting
clang-format --dry-run --Werror src/**/*.cpp include/**/*.h
```

#### Python Formatting

Use `black` for Python code formatting:

```bash
# Format Python code
black python/

# Check formatting
black --check python/
```

## Testing Guidelines

### Test Requirements

All contributions must include appropriate tests:

1. **Unit tests** for new functionality
2. **Integration tests** for component interactions
3. **Performance tests** for performance-critical code
4. **Documentation tests** for examples and usage

### Test Structure

#### Unit Tests

```cpp
TEST(MyClassTest, Constructor) {
    // Arrange
    MyClassConfig config;
    config.value = 42;
    
    // Act
    MyClass instance(config);
    
    // Assert
    EXPECT_TRUE(instance.IsInitialized());
    EXPECT_EQ(instance.GetValue(), 42);
}

TEST(MyClassTest, InvalidInput) {
    // Arrange
    MyClass instance;
    
    // Act & Assert
    EXPECT_THROW(instance.ProcessInput(""), std::invalid_argument);
}
```

#### Integration Tests

```cpp
TEST(IntegrationTest, EndToEndWorkflow) {
    // Set up integration test environment
    IntegrationTestEnvironment env;
    env.SetUp();
    
    // Test complete workflow
    auto result = env.ExecuteWorkflow();
    
    // Verify results
    EXPECT_EQ(result.status, Status::OK);
    EXPECT_GT(result.performance, 0.0);
    
    // Clean up
    env.TearDown();
}
```

#### Performance Tests

```cpp
TEST(PerformanceTest, InferenceLatency) {
    EdgeAIEngine engine(config);
    engine.Initialize();
    engine.LoadModel("test_model.onnx", ModelType::ONNX);
    
    const int num_iterations = 1000;
    std::vector<double> latencies;
    
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        engine.RunInference(inputs, outputs);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        latencies.push_back(duration.count());
    }
    
    // Verify performance requirements
    double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    EXPECT_LT(avg_latency, 10000);  // Less than 10ms average
}
```

### Test Coverage

Maintain high test coverage:

- **Overall coverage**: > 90%
- **Critical components**: > 95%
- **New code**: > 95%

### Running Tests

```bash
# Run all tests
make test

# Run specific test suite
./bin/edge_ai_engine_tests --gtest_filter="MyClassTest.*"

# Run with coverage
make coverage
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   make test
   ```

2. **Check code formatting**:
   ```bash
   clang-format --dry-run --Werror src/**/*.cpp include/**/*.h
   black --check python/
   ```

3. **Run static analysis**:
   ```bash
   cppcheck src/
   flake8 python/
   ```

4. **Update documentation** if needed

5. **Rebase on latest main**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests added/updated
- [ ] All tests pass locally

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance impact assessed
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in CI/CD pipeline
4. **Approval** from at least one maintainer
5. **Merge** by maintainer

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 20.04]
- Compiler: [e.g., GCC 10.3.0]
- Version: [e.g., 1.0.0]

## Additional Context
Any additional information
```

### Feature Requests

Use the feature request template:

```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other approaches considered

## Additional Context
Any additional information
```

## Documentation

### Code Documentation

- **Public APIs** must be documented
- **Complex algorithms** should include comments
- **Examples** should be provided for new features

### User Documentation

- **README updates** for new features
- **API documentation** for new interfaces
- **Tutorial updates** for new workflows

### Documentation Standards

- Use clear, concise language
- Provide examples where helpful
- Keep documentation up to date
- Use consistent formatting

## Development Tools

### Recommended IDE Setup

#### Visual Studio Code

Extensions:
- C/C++ (Microsoft)
- CMake Tools
- Python
- GitLens

Settings:
```json
{
    "C_Cpp.default.cppStandard": "c++20",
    "C_Cpp.default.compilerPath": "/usr/bin/gcc",
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "python.defaultInterpreterPath": "/usr/bin/python3"
}
```

#### CLion

- Configure CMake settings
- Set C++ standard to C++20
- Enable clang-format integration
- Configure Python interpreter

### Pre-commit Hooks

Install pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Pre-commit configuration (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 22.12.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Performance Guidelines

### Performance Requirements

- **New features** should not degrade performance
- **Performance-critical code** must be optimized
- **Benchmarks** should be provided for performance changes

### Profiling

Use the built-in profiler for performance analysis:

```cpp
#include "profiling/profiler.h"

void MyFunction() {
    PROFILER_SCOPED_EVENT("my_function");
    
    // Your code here
}
```

### Memory Management

- **Avoid memory leaks**
- **Use efficient data structures**
- **Minimize allocations** in hot paths
- **Use memory pools** for frequent allocations

## Security Guidelines

### Security Considerations

- **Input validation** for all public APIs
- **Buffer overflow protection**
- **Secure coding practices**
- **Regular security audits**

### Code Review Checklist

- [ ] Input validation implemented
- [ ] No buffer overflows
- [ ] No memory leaks
- [ ] Error handling implemented
- [ ] No hardcoded secrets
- [ ] Proper resource cleanup

## Release Process

### Versioning

Follow semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Release notes prepared
- [ ] Tag created
- [ ] Release published

## Community Guidelines

### Code of Conduct

- **Be respectful** and inclusive
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be collaborative** in discussions

### Communication

- **GitHub Issues** for bug reports and feature requests
- **GitHub Discussions** for questions and general discussion
- **Pull Requests** for code contributions
- **Email** for security issues

### Getting Help

- **Check documentation** first
- **Search existing issues** for similar problems
- **Ask questions** in GitHub Discussions
- **Join community** discussions

## License

By contributing to the Edge AI Inference Engine, you agree that your contributions will be licensed under the MIT License.
