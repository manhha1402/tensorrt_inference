#pragma once
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "tensorrt_inference/paddle_ocr/clipper.h"
#include "tensorrt_inference/paddle_ocr/paddleocr_utils.h"
#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"

namespace tensorrt_inference
{

  class TextDetection : public Engine
  {
  public:
    explicit TextDetection(
        const std::string &model_name,
        tensorrt_inference::Options options = tensorrt_inference::Options(),
        const std::filesystem::path &model_dir =
            std::filesystem::path(std::getenv("HOME")) / "data" / "weights");
    uint32_t getMaxOutputLength(const nvinfer1::Dims &tensorShape) const override;

    void runInference(const cv::cuda::GpuMat &gpu_img,
                      std::vector<std::vector<std::vector<int>>> &boxes);

  private:
    std::string onnx_file_;
    int max_side_len_ = 960;
    std::array<float, 3> mean_ = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    double det_db_thresh_ = 0.3;
    double det_db_box_thresh_ = 0.5;
    double det_db_unclip_ratio_ = 2.0;
    bool use_polygon_score_ = false;
  };

  class TextRecognition : public Engine
  {
  public:
    explicit TextRecognition(
        const std::string &model_name,
        tensorrt_inference::Options options = tensorrt_inference::Options(),
        const std::filesystem::path &model_dir =
            std::filesystem::path(std::getenv("HOME")) / "data" / "weights");
    uint32_t getMaxOutputLength(const nvinfer1::Dims &tensorShape) const override;

    std::pair<std::string, double> runInference(
        std::vector<cv::cuda::GpuMat> &img_list);
    std::vector<std::string> label_list_;

    template <class ForwardIterator>
    inline static size_t argmax(ForwardIterator first, ForwardIterator last)
    {
      return std::distance(first, std::max_element(first, last));
    }

  protected:
    std::string onnx_file_;

    std::array<float, 3> mean_ = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    int rec_batch_num_ = 6;
  };

  class PaddleOCR
  {
  public:
    explicit PaddleOCR(
        const std::string &model_name,
        tensorrt_inference::Options options_det = tensorrt_inference::Options(),
        tensorrt_inference::Options options_rec = tensorrt_inference::Options(),
        const std::filesystem::path &model_dir =
            std::filesystem::path(std::getenv("HOME")) / "data" / "weights");

    std::pair<std::string, double> runInference(
        cv::Mat &img, const CroppedObject &cropped_plate);

    cv::Mat drawBBoxLabels(const cv::Mat &image,
                           const std::vector<CroppedObject> &objects,
                           unsigned int scale = 2);

  private:
    std::shared_ptr<TextDetection> text_det_;
    std::shared_ptr<TextRecognition> text_rec_;
  };

} // namespace tensorrt_inference