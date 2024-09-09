#pragma once 
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"
#include <yaml-cpp/yaml.h>
#include <tensorrt_inference/model.h>

namespace tensorrt_inference
{

class Detection : public Model 
{
public:
    explicit Detection(const std::string& model_dir,const YAML::Node &config);

   // Detect the objects in the image
    std::vector<Object> detectObjects(const cv::Mat &inputImageBGR);
    std::vector<Object> detectObjects(cv::cuda::GpuMat &inputImageBGR);
    void drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale = 2);
protected:
    virtual std::vector<Object> postprocess(std::unordered_map<std::string, std::vector<float>> &feature_vector) = 0;
    std::map<int, std::string> class_labels_;
    int CATEGORY;
    bool agnostic_;
    std::vector<cv::Scalar> class_colors_;
};
}