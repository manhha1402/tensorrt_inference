#pragma once 
#include <fstream>
#include <opencv2/opencv.hpp>

// Utility method for checking if a file exists on disk
inline bool doesFileExist(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

struct Object {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Pose estimation key points
    std::vector<float> kps{};
};