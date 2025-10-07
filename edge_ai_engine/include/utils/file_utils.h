/**
 * @file file_utils.h
 * @brief File utility interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the FileUtils class for file operations in the Edge AI Engine.
 */

#pragma once

#include <string>
#include <vector>

namespace EdgeAI {

/**
 * @class FileUtils
 * @brief File utility class
 * 
 * The FileUtils class provides utility functions for file operations.
 */
class FileUtils {
public:
    /**
     * @brief Constructor
     */
    FileUtils();
    
    /**
     * @brief Destructor
     */
    ~FileUtils();
    
    // Disable copy constructor and assignment operator
    FileUtils(const FileUtils&) = delete;
    FileUtils& operator=(const FileUtils&) = delete;
    
    /**
     * @brief Check if file exists
     * @param file_path Path to file
     * @return True if file exists
     */
    static bool FileExists(const std::string& file_path);
    
    /**
     * @brief Check if directory exists
     * @param dir_path Path to directory
     * @return True if directory exists
     */
    static bool DirectoryExists(const std::string& dir_path);
    
    /**
     * @brief Get file size
     * @param file_path Path to file
     * @return File size in bytes
     */
    static size_t GetFileSize(const std::string& file_path);
    
    /**
     * @brief Get file extension
     * @param file_path Path to file
     * @return File extension
     */
    static std::string GetFileExtension(const std::string& file_path);
    
    /**
     * @brief Get file name
     * @param file_path Path to file
     * @return File name
     */
    static std::string GetFileName(const std::string& file_path);
    
    /**
     * @brief Get directory path
     * @param file_path Path to file
     * @return Directory path
     */
    static std::string GetDirectoryPath(const std::string& file_path);
    
    /**
     * @brief Create directory
     * @param dir_path Path to directory
     * @return True if directory created successfully
     */
    static bool CreateDirectory(const std::string& dir_path);
    
    /**
     * @brief Delete file
     * @param file_path Path to file
     * @return True if file deleted successfully
     */
    static bool DeleteFile(const std::string& file_path);
    
    /**
     * @brief Delete directory
     * @param dir_path Path to directory
     * @return True if directory deleted successfully
     */
    static bool DeleteDirectory(const std::string& dir_path);
    
    /**
     * @brief Copy file
     * @param src_path Source file path
     * @param dst_path Destination file path
     * @return True if file copied successfully
     */
    static bool CopyFile(const std::string& src_path, const std::string& dst_path);
    
    /**
     * @brief Move file
     * @param src_path Source file path
     * @param dst_path Destination file path
     * @return True if file moved successfully
     */
    static bool MoveFile(const std::string& src_path, const std::string& dst_path);
    
    /**
     * @brief Read binary file
     * @param file_path Path to file
     * @return Binary data
     */
    static std::vector<uint8_t> ReadBinaryFile(const std::string& file_path);
    
    /**
     * @brief Write binary file
     * @param file_path Path to file
     * @param data Binary data to write
     * @return True if file written successfully
     */
    static bool WriteBinaryFile(const std::string& file_path, const std::vector<uint8_t>& data);
    
    /**
     * @brief Read text file
     * @param file_path Path to file
     * @return Text content
     */
    static std::string ReadTextFile(const std::string& file_path);
    
    /**
     * @brief Write text file
     * @param file_path Path to file
     * @param content Text content to write
     * @return True if file written successfully
     */
    static bool WriteTextFile(const std::string& file_path, const std::string& content);
};

} // namespace EdgeAI
