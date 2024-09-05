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
    std::vector<Object> detectObjects(const cv::cuda::GpuMat &inputImageBGR);
    void drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale = 2);
protected:
    virtual std::vector<Object> postprocessDetect(std::vector<float> &feature_vector) = 0;
    std::map<int, std::string> class_labels_;
    int CATEGORY;
    float obj_threshold_;
    float nms_threshold_;
    bool agnostic_;
    std::vector<cv::Scalar> class_colors_;
    std::vector<int> strides_;
    std::vector<int> num_anchors_;
    int num_rows_ = 0;
};
}