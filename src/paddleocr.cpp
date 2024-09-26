#include "tensorrt_inference/paddleocr.h"

#include <opencv2/cudaimgproc.hpp>
namespace tensorrt_inference {
PaddleOCR::PaddleOCR(const std::string &model_name,
                     tensorrt_inference::Options options,
                     const std::filesystem::path &model_dir)
    : Engine(options) {
  std::string config_file = (model_dir / model_name / "config.yaml").string();
  YAML::Node config = YAML::LoadFile(config_file);

  onnx_file_ =
      (model_dir / model_name / config["onnx_file"].as<std::string>()).string();
  std::cout << onnx_file_ << std::endl;
  // Specify options for GPU inference
  // options.engine_file_dir = getFolderOfFile(onnx_file_);
  // std::cout << options.engine_file_dir << std::endl;
  if (config["num_kps"]) {
    num_kps_ = config["num_kps"].as<int>();
  }
  if (config["normalized"]) {
    normalized_ = config["normalized"].as<bool>();
  }
  if (config["swapBR"]) {
    swapBR_ = config["swapBR"].as<bool>();
  }
  if (config["sub_vals"]) {
    sub_vals_ = config["sub_vals"].as<std::vector<float>>();
  }
  if (config["div_vals"]) {
    div_vals_ = config["div_vals"].as<std::vector<float>>();
  }

  if (config["labels_file"]) {
    std::string labels_file =
        (model_dir / model_name / config["labels_file"].as<std::string>())
            .string();
    if (Util::doesFileExist(std::filesystem::path(labels_file))) {
      class_labels_ = readClassLabel(labels_file);
    } else {
      spdlog::error("label file is not existed!");
    }
  }
  auto succ = buildLoadNetwork(onnx_file_);
  if (!succ) {
    const std::string errMsg =
        "Error: Unable to build or load the TensorRT engine. "
        "Try increasing TensorRT log severity to kVERBOSE (in "
        "/libs/tensorrt-cpp-api/engine.cpp).";
    throw std::runtime_error(errMsg);
  }
}
uint32_t PaddleOCR::getMaxOutputLength(
    const nvinfer1::Dims &tensorShape) const {
  return (m_options.MAX_DIMS_[3] + 4) / 8 *
         tensorShape.d[tensorShape.nbDims - 1] * rec_batch_num_;
}

cv::cuda::GpuMat PaddleOCR::preprocess(const cv::cuda::GpuMat &gpuImg) {
  // Populate the input vectors
  const auto &input_info = getInputInfo().begin();

  // These params will be used in the post-processing stage
  input_frame_h_ = gpuImg.rows;
  input_frame_w_ = gpuImg.cols;
  m_ratio =
      1.f /
      std::min(input_info->second.dims.d[3] / static_cast<float>(gpuImg.cols),
               input_info->second.dims.d[2] / static_cast<float>(gpuImg.rows));
  // Convert the image from BGR to RGB
  cv::cuda::GpuMat rgbMat = gpuImg;
  auto resized = rgbMat;
  // Resize to the model expected input size while maintaining the aspect ratio
  // with the use of padding
  if (resized.rows != input_info->second.dims.d[2] ||
      resized.cols != input_info->second.dims.d[3]) {
    // Only resize if not already the right size to avoid unecessary copy
    resized = resizeKeepAspectRatioPadRightBottom(
        rgbMat, input_info->second.dims.d[2], input_info->second.dims.d[3],
        cv::Scalar(128, 128, 128));
  }
  cv::cuda::GpuMat gpu_dst(1, resized.rows * resized.cols, CV_8UC3);
  size_t width = resized.cols * resized.rows;
  if (swapBR_) {
    std::vector<cv::cuda::GpuMat> input_channels{
        cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
                         &(gpu_dst.ptr()[width * 2])),
        cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
                         &(gpu_dst.ptr()[width])),
        cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
                         &(gpu_dst.ptr()[0]))};
    cv::cuda::split(resized, input_channels);  // HWC -> CHW
  } else {
    std::vector<cv::cuda::GpuMat> input_channels{
        cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
                         &(gpu_dst.ptr()[0])),
        cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
                         &(gpu_dst.ptr()[width])),
        cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
                         &(gpu_dst.ptr()[width * 2]))};
    cv::cuda::split(resized, input_channels);  // HWC -> CHW
  }
  cv::cuda::GpuMat mfloat;
  if (normalized_) {
    // [0.f, 1.f]
    gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
    for (auto &val : sub_vals_) val = val / 255.0f;
  } else {
    // [0.f, 255.f]
    gpu_dst.convertTo(mfloat, CV_32FC3);
  }

  // Apply scaling and mean subtraction
  cv::cuda::subtract(mfloat,
                     cv::Scalar(sub_vals_[0], sub_vals_[1], sub_vals_[2]),
                     mfloat, cv::noArray(), -1);
  cv::cuda::divide(mfloat, cv::Scalar(div_vals_[0], div_vals_[1], div_vals_[2]),
                   mfloat, 1, -1);
  return mfloat;
}

}  // namespace tensorrt_inference