#pragma once

#include <filesystem>
namespace tensorrt_inference
{
    namespace Util
    {

        inline bool doesFileExist(const std::string &filepath)
        {
            std::ifstream f(filepath.c_str());
            return f.good();
        }

        inline void checkCudaErrorCode(cudaError_t code)
        {
            if (code != cudaSuccess)
            {
                std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + " (" + cudaGetErrorName(code) +
                                     "), with message: " + cudaGetErrorString(code);
                spdlog::error(errMsg);
                throw std::runtime_error(errMsg);
            }
        }

        inline std::vector<std::string> getFilesInDirectory(const std::string &dirPath)
        {
            std::vector<std::string> fileNames;
            for (const auto &entry : std::filesystem::directory_iterator(dirPath))
            {
                if (entry.is_regular_file())
                {
                    fileNames.push_back(entry.path().string());
                }
            }
            return fileNames;
        }
    }
    inline std::filesystem::path getFolderOfFile(const std::string &filepath)
    {
        std::filesystem::path pathObj(filepath);

        // Check if the provided path exists and is a file
        if (std::filesystem::exists(pathObj) && std::filesystem::is_regular_file(pathObj))
        {
            // Return the parent directory of the file
            return pathObj.parent_path();
        }
        // Return an empty string if the path does not exist or is not a file
        return std::filesystem::path("");
    }
}