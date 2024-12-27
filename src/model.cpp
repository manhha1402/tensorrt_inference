#include <tensorrt_inference/model.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
namespace tensorrt_inference
{
  Model::Model(const std::string &model_name, tensorrt_inference::Options options,
               const std::filesystem::path &model_dir)
  {
    onnx_file_ = (model_dir / model_name / (model_name + ".onnx")).string();
    if (!std::filesystem::exists(onnx_file_))
    {
      throw std::runtime_error("onnx file not existed");
    }
    setDefaultParams(model_name);

    // Specify options for GPU inference
    options.engine_file_dir = getFolderOfFile(onnx_file_);
    m_trtEngine = std::make_unique<Engine>(options);
    auto succ = m_trtEngine->buildLoadNetwork(onnx_file_);
    if (!succ)
    {
      const std::string errMsg =
          "Error: Unable to build or load the TensorRT engine. "
          "Try increasing TensorRT log severity to kVERBOSE (in "
          "/libs/tensorrt-cpp-api/engine.cpp).";
      throw std::runtime_error(errMsg);
    }
  }
  Model::~Model() {}

  void Model::setDefaultParams(const std::string &model_name)
  {
    if (model_name.find("facedetector") != std::string::npos || model_name.find("retinaface") != std::string::npos)
    {
      preprocess_params_ = PreprocessParams(cv::Scalar(104, 117, 123), cv::Scalar(1.0, 1.0, 1.0), false, true, true);
    }
    else if (model_name.find("yolo") != std::string::npos)
    {
      preprocess_params_ = PreprocessParams(cv::Scalar(0, 0, 0), cv::Scalar(1.0, 1.0, 1.0), true, true, true);
    }
    else
    {
      preprocess_params_ = PreprocessParams();
    }
    preprocess_params_.printInfo();
  }
  void Model::setParams(const PreprocessParams &params)
  {
    preprocess_params_ = params;
    preprocess_params_.printInfo();
  }
  bool Model::preProcess(const cv::Mat &img)
  {
    cv::Mat mat;
    if (preprocess_params_.swapBR)
    {
      cv::cvtColor(img, mat, cv::COLOR_BGR2RGB);
    }
    else
    {
      mat = img.clone();
    }

    const auto &input_info = m_trtEngine->getInputInfo().begin();
    input_frame_w_ = float(img.cols);
    input_frame_h_ = float(img.rows);
    std::vector<float> input_buff(input_info->second.tensor_length);
    cv::Mat resized_img;
    if (preprocess_params_.keep_ratio)
    {
      ratios_[0] = ratios_[1] = std::max(float(img.cols) / float(input_info->second.dims.d[3]), float(img.rows) / float(input_info->second.dims.d[3]));
      float resize_ratio = std::min(float(input_info->second.dims.d[3]) / float(img.cols), float(input_info->second.dims.d[2]) / float(img.rows));
      resized_img = cv::Mat::zeros(cv::Size(input_info->second.dims.d[3], input_info->second.dims.d[2]), CV_8UC3);
      cv::Mat rsz_img;
      cv::resize(mat, rsz_img, cv::Size(), resize_ratio, resize_ratio);
      rsz_img.copyTo(resized_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    }
    else
    {
      ratios_[0] = float(img.cols) / float(input_info->second.dims.d[3]);
      ratios_[1] = float(img.rows) / float(input_info->second.dims.d[2]);
      cv::resize(mat, resized_img, cv::Size(input_info->second.dims.d[3], input_info->second.dims.d[2]));
    }
    if (preprocess_params_.normalized)
    {
      resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);
      preprocess_params_.sub_vals = preprocess_params_.sub_vals / 255.0;
    }
    else
    {
      resized_img.convertTo(resized_img, CV_32FC3);
    }
    // pepform substraction
    cv::subtract(resized_img, preprocess_params_.sub_vals, resized_img, cv::noArray(), -1);
    // perfrom divide
    cv::divide(resized_img, preprocess_params_.div_vals, resized_img);

    for (int i = 0; i < resized_img.channels(); ++i)
    {
      cv::extractChannel(resized_img, cv::Mat(resized_img.rows, resized_img.cols, CV_32FC1, input_buff.data() + i * resized_img.rows * resized_img.cols), i);
    }
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
    Util::checkCudaErrorCode(
        cudaMemcpyAsync(m_trtEngine->input_map_.begin()->second.buffer, input_buff.data(),
                        m_trtEngine->input_map_.begin()->second.tensor_length,
                        cudaMemcpyHostToDevice, inferenceCudaStream));

    return true;
  }

  bool Model::doInference(
      const cv::Mat &img,
      std::unordered_map<std::string, std::vector<float>> &feature_vectors)
  {
    bool res = preProcess(img);
    std::cout << "preprocess done" << std::endl;
    auto succ = m_trtEngine->runInference(feature_vectors);
    std::cout << "runInference done" << std::endl;

    if (!succ)
    {
      spdlog::error("Error: Unable to run inference.");
      return false;
    }
    return true;
  }

} // namespace tensorrt_inference