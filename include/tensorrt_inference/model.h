
#pragma once

#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"
namespace tensorrt_inference {

struct DetectionParams {
  float obj_threshold;
  float nms_threshold;
  float seg_threshold;
  float kps_threshold;
  int num_detect;

  DetectionParams()
      : obj_threshold(0.25f),
        nms_threshold(0.65f),
        seg_threshold(0.5),
        kps_threshold(0.5),
        num_detect(20) {}
  DetectionParams(const float obj_threshold, const float nms_threshold,
                  const float seg_threshold, const float kps_threshold,
                  const int num_detect)
      : obj_threshold(obj_threshold),
        nms_threshold(nms_threshold),
        seg_threshold(seg_threshold),
        kps_threshold(kps_threshold),
        num_detect(num_detect) {}
};

class Model {
 public:
  explicit Model(const std::string &model_name,
                 const std::filesystem::path &model_dir =
                     std::filesystem::path(std::getenv("HOME")) / "data" /
                     "weights");
  ~Model();
  bool doInference(
      cv::cuda::GpuMat &gpuImg,
      std::unordered_map<std::string, std::vector<float>> &feature_vectors);
  std::unique_ptr<Engine<float>> m_trtEngine = nullptr;

 protected:
  // Preprocess the input. Normalize values between [0.f, 1.f] Setting the
  // normalize flag to false will leave values between [0.f, 255.f] (some
  // converted models may require this). If the model requires values to be
  // normalized between [-1.f, 1.f], use the following params:
  //    subVals = {0.0f, 0.0f, 0.0f};
  //    divVals = {1.f, 1.f, 1.f};
  //    normalize = false;
  cv::cuda::GpuMat preprocess(const cv::cuda::GpuMat &gpuImg);
  std::string onnx_file_;

  float m_ratio = 1.0;
  float input_frame_h_ = 0;
  float input_frame_w_ = 0;
  std::vector<float> sub_vals_{0, 0, 0};
  std::vector<float> div_vals_{1.0f, 1.0f, 1.0f};
  bool normalized_ = false;
  bool swapBR_ = true;
  int num_kps_ = 17;
};
}  // namespace tensorrt_inference
