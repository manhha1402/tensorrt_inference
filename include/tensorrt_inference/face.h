#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"
#include <yaml-cpp/yaml.h>
#include <tensorrt_inference/model.h>
#define CLIP(a, min, max) (MAX(MIN(a, max), min)) // MIN, MAX defined in opencv

namespace tensorrt_inference
{
struct anchorBox {
    float cx;
    float cy;
    float sx;
    float sy;
};

class Face : public Model
{
public:
    explicit Face(const std::string& model_dir,const YAML::Node &config);
   // Detect the objects in the image
    std::vector<FaceBox> detectFaces(const cv::Mat &inputImageBGR,const ModelParams& params = ModelParams(0.2,0.2,20));
    std::vector<FaceBox> detectFaces(cv::cuda::GpuMat &inputImageBGR,const ModelParams& params = ModelParams(0.2,0.2,20));

    cv::Mat drawFaceLabels(const cv::Mat &image, const std::vector<FaceBox> &faces, unsigned int scale=2);

protected:
    virtual std::vector<FaceBox> postProcess(std::unordered_map<std::string, std::vector<float>> &feature_vectors,const ModelParams& params = ModelParams(0.2,0.2,20)) = 0;
    virtual void create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h) = 0 ;
    float landmark_std_;
};
}
