/**
 * @file file_utils.cpp
 * @brief File utility implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "utils/file_utils.h"
#include "utils/logger.h"
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>

namespace EdgeAI {

FileUtils::FileUtils() = default;

FileUtils::~FileUtils() = default;

bool FileUtils::FileExists(const std::string& file_path) {
    struct stat buffer;
    return (stat(file_path.c_str(), &buffer) == 0);
}

bool FileUtils::DirectoryExists(const std::string& dir_path) {
    struct stat buffer;
    return (stat(dir_path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

size_t FileUtils::GetFileSize(const std::string& file_path) {
    struct stat buffer;
    if (stat(file_path.c_str(), &buffer) == 0) {
        return buffer.st_size;
    }
    return 0;
}

std::string FileUtils::GetFileExtension(const std::string& file_path) {
    size_t pos = file_path.find_last_of('.');
    if (pos != std::string::npos) {
        return file_path.substr(pos + 1);
    }
    return "";
}

std::string FileUtils::GetFileName(const std::string& file_path) {
    size_t pos = file_path.find_last_of('/');
    if (pos != std::string::npos) {
        return file_path.substr(pos + 1);
    }
    return file_path;
}

std::string FileUtils::GetDirectoryPath(const std::string& file_path) {
    size_t pos = file_path.find_last_of('/');
    if (pos != std::string::npos) {
        return file_path.substr(0, pos);
    }
    return "";
}

bool FileUtils::CreateDirectory(const std::string& dir_path) {
    try {
        if (DirectoryExists(dir_path)) {
            return true;
        }
        
        // Create parent directories if they don't exist
        std::string parent_dir = GetDirectoryPath(dir_path);
        if (!parent_dir.empty() && !DirectoryExists(parent_dir)) {
            if (!CreateDirectory(parent_dir)) {
                return false;
            }
        }
        
        // Create the directory
        return mkdir(dir_path.c_str(), 0755) == 0;
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error creating directory: " + std::string(e.what()));
        return false;
    }
}

bool FileUtils::DeleteFile(const std::string& file_path) {
    try {
        if (!FileExists(file_path)) {
            return true;
        }
        
        return unlink(file_path.c_str()) == 0;
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error deleting file: " + std::string(e.what()));
        return false;
    }
}

bool FileUtils::DeleteDirectory(const std::string& dir_path) {
    try {
        if (!DirectoryExists(dir_path)) {
            return true;
        }
        
        return rmdir(dir_path.c_str()) == 0;
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error deleting directory: " + std::string(e.what()));
        return false;
    }
}

bool FileUtils::CopyFile(const std::string& src_path, const std::string& dst_path) {
    try {
        std::ifstream src(src_path, std::ios::binary);
        if (!src.is_open()) {
            Logger::GetInstance().Error("Failed to open source file: " + src_path);
            return false;
        }
        
        std::ofstream dst(dst_path, std::ios::binary);
        if (!dst.is_open()) {
            Logger::GetInstance().Error("Failed to create destination file: " + dst_path);
            return false;
        }
        
        dst << src.rdbuf();
        
        src.close();
        dst.close();
        
        return true;
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error copying file: " + std::string(e.what()));
        return false;
    }
}

bool FileUtils::MoveFile(const std::string& src_path, const std::string& dst_path) {
    try {
        if (!FileExists(src_path)) {
            Logger::GetInstance().Error("Source file does not exist: " + src_path);
            return false;
        }
        
        // Try to rename first (atomic operation)
        if (rename(src_path.c_str(), dst_path.c_str()) == 0) {
            return true;
        }
        
        // Fallback to copy and delete
        if (CopyFile(src_path, dst_path)) {
            return DeleteFile(src_path);
        }
        
        return false;
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error moving file: " + std::string(e.what()));
        return false;
    }
}

std::vector<uint8_t> FileUtils::ReadBinaryFile(const std::string& file_path) {
    std::vector<uint8_t> data;
    
    try {
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            Logger::GetInstance().Error("Failed to open file: " + file_path);
            return data;
        }
        
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        data.resize(size);
        file.read(reinterpret_cast<char*>(data.data()), size);
        
        file.close();
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error reading binary file: " + std::string(e.what()));
        data.clear();
    }
    
    return data;
}

bool FileUtils::WriteBinaryFile(const std::string& file_path, const std::vector<uint8_t>& data) {
    try {
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            Logger::GetInstance().Error("Failed to create file: " + file_path);
            return false;
        }
        
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
        file.close();
        
        return true;
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error writing binary file: " + std::string(e.what()));
        return false;
    }
}

std::string FileUtils::ReadTextFile(const std::string& file_path) {
    std::string content;
    
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            Logger::GetInstance().Error("Failed to open file: " + file_path);
            return content;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        content = buffer.str();
        
        file.close();
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error reading text file: " + std::string(e.what()));
        content.clear();
    }
    
    return content;
}

bool FileUtils::WriteTextFile(const std::string& file_path, const std::string& content) {
    try {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            Logger::GetInstance().Error("Failed to create file: " + file_path);
            return false;
        }
        
        file << content;
        file.close();
        
        return true;
    } catch (const std::exception& e) {
        Logger::GetInstance().Error("Error writing text file: " + std::string(e.what()));
        return false;
    }
}

} // namespace EdgeAI
