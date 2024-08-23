#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"
#include <yaml-cpp/yaml.h>
#include <tensorrt_inference/model.h>
namespace tensorrt_inference
{
class Face : public Model
{
public:
    explicit Face(const std::string& model_dir,const YAML::Node &config);
   // Detect the objects in the image
    std::vector<FaceBox> detectFaces(const cv::Mat &inputImageBGR);
    std::vector<FaceBox> detectFaces(const cv::cuda::GpuMat &inputImageBGR);
    void drawFaceLabels(cv::Mat &image, const std::vector<FaceBox> &faces, unsigned int scale = 2);

protected:
    virtual std::vector<FaceBox> postProcess(std::vector<float> &feature_vector) = 0;
    virtual void generateAnchors() = 0;
    float obj_threshold_;
    float nms_threshold_;
    bool detect_mask_;
    float mask_thresh_;
    float landmark_std_;

    int bbox_head_ = 3;
    int landmark_head_ = 10;
    std::vector<int> feature_sizes_;
    std::vector<int> feature_steps_;
    std::vector<std::vector<int>> feature_maps_;
    int sum_of_feature_;
};
}
