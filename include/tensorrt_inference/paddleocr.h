#pragma once
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"

namespace tensorrt_inference {

class PaddleOCR : public Engine<float> {
 public:
  explicit PaddleOCR(
      const std::string &model_name,
      tensorrt_inference::Options options = tensorrt_inference::Options(),
      const std::filesystem::path &model_dir =
          std::filesystem::path(std::getenv("HOME")) / "data" / "weights");
  uint32_t getMaxOutputLength(const nvinfer1::Dims &tensorShape) const override;
  cv::cuda::GpuMat preprocess(const cv::cuda::GpuMat &gpuImg);

 protected:
  std::map<int, std::string> class_labels_;
  std::string onnx_file_;

  float m_ratio = 1.0;
  float input_frame_h_ = 0;
  float input_frame_w_ = 0;
  std::vector<float> sub_vals_{0, 0, 0};
  std::vector<float> div_vals_{1.0f, 1.0f, 1.0f};
  bool normalized_ = false;
  bool swapBR_ = true;
  int num_kps_ = 17;
  int rec_batch_num_ = 6;
};
}  // namespace tensorrt_inference