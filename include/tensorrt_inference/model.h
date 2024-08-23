
#pragma once

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"
namespace tensorrt_inference
{
class Model
{
public:
    explicit Model(const std::string& model_dir,const YAML::Node &config);
    ~Model();
    bool doInference(const cv::cuda::GpuMat &gpuImg,  std::vector<std::vector<std::vector<float>>>& feature_vectors); 

protected:
    std::unique_ptr<Engine<float>> m_trtEngine = nullptr;
    // Preprocess the input
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg);
    std::string onnx_file_;
    std::string labels_file_;

    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;
};
}
