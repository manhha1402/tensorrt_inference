#pragma once 
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"
#include <yaml-cpp/yaml.h>
#include <tensorrt_inference/model.h>

namespace tensorrt_inference
{
struct DetectionParams
{
    ModelParams params;
    float seg_threshold;
    float kps_threshold;
    DetectionParams(): seg_threshold(0.5), kps_threshold(0.5)
    {}
};
class Detection : public Model 
{
public:
    explicit Detection(const std::string& model_dir,const YAML::Node &config);

   // Detect the objects in the image
    std::vector<Object> detectObjects(const cv::Mat &inputImageBGR, const DetectionParams& params = DetectionParams());
    std::vector<Object> detectObjects(cv::cuda::GpuMat &inputImageBGR, const DetectionParams& params = DetectionParams());
    void drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, const DetectionParams& params = DetectionParams(),const std::vector<std::string>& detected_class = {}, unsigned int scale = 2);
      // Draw the object bounding boxes and labels on the image
    void drawBBoxLabel(cv::Mat &image, const Object &object, const DetectionParams& params = DetectionParams(), unsigned int scale = 2);
    void drawSegmentation(cv::Mat &mask, const Object &object);


protected:
    virtual std::vector<Object> postprocess(std::unordered_map<std::string, std::vector<float>> &feature_vector, const DetectionParams& params = DetectionParams()) = 0;
    std::map<int, std::string> class_labels_;
    int CATEGORY;
    bool agnostic_;
    int num_kps_ = 17;
    std::vector<cv::Scalar> class_colors_;

    const std::vector<std::vector<unsigned int>> KPS_COLORS = {
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},   {255, 128, 0},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51, 153, 255},
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};

    const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
                                                             {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
                                                             {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};
};
}