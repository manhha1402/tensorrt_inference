#pragma once 
#include <fstream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
namespace tensorrt_inference
{
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
    cv::Mat box_mask;
    // Pose estimation key points
    std::vector<float> kps{};
    //std::vector<cv::Point2f> kps;
};

// struct KeyPoint{
//     int x;
//     int y;
// };
// struct FaceBox{
//     // The object bounding box rectangle.
//     cv::Rect_<float> rect;
//     std::vector<KeyPoint> landmarks;
//     bool has_mask = false;
//     float score = 0;
// };



inline std::map<int, std::string> readClassLabel(const std::string &fileName)
{
    YAML::Node config = YAML::LoadFile(fileName);

    std::map<int, std::string> class_label;
    for (size_t i = 0; i < config.size(); ++i) {
        class_label[i] = config[i].as<std::string>();
    }
    return class_label;
}


}