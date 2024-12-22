#include <tensorrt_inference/model.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
namespace tensorrt_inference
{
  Model::Model(const std::string &model_name, tensorrt_inference::Options options,
               const std::filesystem::path &model_dir)
  {
    std::string config_file = (model_dir / model_name / "config.yaml").string();
    YAML::Node config = YAML::LoadFile(config_file);

    onnx_file_ =
        (model_dir / model_name / config["onnx_file"].as<std::string>()).string();
    if (config["num_kps"])
    {
      num_kps_ = config["num_kps"].as<int>();
    }
    if (config["normalized"])
    {
      normalized_ = config["normalized"].as<bool>();
    }
    if (config["swapBR"])
    {
      swapBR_ = config["swapBR"].as<bool>();
    }
    if (config["sub_vals"])
    {
      sub_vals_ = config["sub_vals"].as<std::vector<float>>();
    }
    if (config["div_vals"])
    {
      div_vals_ = config["div_vals"].as<std::vector<float>>();
    }
    if (config["keep_ratio"])
    {
      keep_ratio_ = config["keep_ratio"].as<bool>();
    }
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

  bool Model::preProcess(const cv::Mat &img)
  {
    cv::Mat mat;
    cv::cvtColor(img, mat, cv::COLOR_BGR2RGB);

    const auto &input_info = m_trtEngine->getInputInfo().begin();
    input_frame_w_ = float(img.cols);
    input_frame_h_ = float(img.rows);
    std::vector<float> input_buff(input_info->second.tensor_length);
    cv::Mat resizeImg;
    if (keep_ratio_)
    {
      ratios_[0] = ratios_[1] = std::max(float(img.cols) / float(input_info->second.dims.d[3]), float(img.rows) / float(input_info->second.dims.d[3]));
      float resize_ratio = std::min(float(input_info->second.dims.d[3]) / float(img.cols), float(input_info->second.dims.d[2]) / float(img.rows));
      resizeImg = cv::Mat::zeros(cv::Size(input_info->second.dims.d[3], input_info->second.dims.d[2]), CV_8UC3);
      cv::Mat rsz_img;
      cv::resize(mat, rsz_img, cv::Size(), resize_ratio, resize_ratio);
      rsz_img.copyTo(resizeImg(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    }
    else
    {
      ratios_[0] = float(img.cols) / float(input_info->second.dims.d[3]);
      ratios_[1] = float(img.rows) / float(input_info->second.dims.d[2]);
      cv::resize(mat, resizeImg, cv::Size(input_info->second.dims.d[3], input_info->second.dims.d[2]));
    }
    // cv::imwrite("resizeImg.jpg", resizeImg);
    // resizeImg.convertTo(resizeImg, CV_32F);
    // sub_vals_ = {104, 117, 123};
    // resizeImg = resizeImg -  cv::Scalar(104, 117, 123);
    resizeImg.convertTo(resizeImg, CV_32FC3);

    cv::imwrite("resizeImg1.jpg", resizeImg);
   // cv::subtract(resizeImg, cv::Scalar(sub_vals_[0],sub_vals_[1],sub_vals_[2]) , resizeImg, cv::noArray(), -1);

    cv::imwrite("resizeImg2.jpg", resizeImg);

    for (int i = 0; i < resizeImg.channels(); ++i)
    {
      cv::extractChannel(resizeImg, cv::Mat(resizeImg.rows, resizeImg.cols, CV_32FC1, input_buff.data() + i * resizeImg.rows * resizeImg.cols), i);
    }
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
    Util::checkCudaErrorCode(
        cudaMemcpyAsync(m_trtEngine->input_map_.begin()->second.buffer, input_buff.data(),
                        m_trtEngine->input_map_.begin()->second.tensor_length,
                        cudaMemcpyHostToDevice, inferenceCudaStream));

    return true;
  }

  // cv::cuda::GpuMat Model::preprocess(const cv::cuda::GpuMat &gpuImg) {
  //   // Populate the input vectors
  //   const auto &input_info = m_trtEngine->getInputInfo().begin();

  //   // These params will be used in the post-processing stage
  //   input_frame_h_ = gpuImg.rows;
  //   input_frame_w_ = gpuImg.cols;

  //   m_ratio =
  //       1.f /
  //       std::min(input_info->second.dims.d[3] / static_cast<float>(gpuImg.cols),
  //                input_info->second.dims.d[2] / static_cast<float>(gpuImg.rows));
  //   // Convert the image from BGR to RGB
  //   cv::cuda::GpuMat rgbMat = gpuImg;
  //   auto resized = rgbMat;
  //   // Resize to the model expected input size while maintaining the aspect ratio
  //   // with the use of padding
  //   if (resized.rows != input_info->second.dims.d[2] ||
  //       resized.cols != input_info->second.dims.d[3]) {
  //     // Only resize if not already the right size to avoid unecessary copy
  //     resized = Engine::resizeKeepAspectRatioPadRightBottom(
  //         rgbMat, input_info->second.dims.d[2], input_info->second.dims.d[3],
  //         cv::Scalar(128, 128, 128));
  //   }
  //   cv::cuda::GpuMat gpu_dst(1, resized.rows * resized.cols, CV_8UC3);
  //   size_t width = resized.cols * resized.rows;
  //   if (swapBR_) {
  //     std::vector<cv::cuda::GpuMat> input_channels{
  //         cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
  //                          &(gpu_dst.ptr()[width * 2])),
  //         cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
  //                          &(gpu_dst.ptr()[width])),
  //         cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
  //                          &(gpu_dst.ptr()[0]))};
  //     cv::cuda::split(resized, input_channels);  // HWC -> CHW
  //   } else {
  //     std::vector<cv::cuda::GpuMat> input_channels{
  //         cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
  //                          &(gpu_dst.ptr()[0])),
  //         cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
  //                          &(gpu_dst.ptr()[width])),
  //         cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U,
  //                          &(gpu_dst.ptr()[width * 2]))};
  //     cv::cuda::split(resized, input_channels);  // HWC -> CHW
  //   }

  //   cv::cuda::GpuMat mfloat;
  //   if (normalized_) {
  //     // [0.f, 1.f]
  //     gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
  //     for (auto &val : sub_vals_) val = val / 255.0f;
  //   } else {
  //     // [0.f, 255.f]
  //     gpu_dst.convertTo(mfloat, CV_32FC3);
  //   }

  //   // Apply scaling and mean subtraction
  //   cv::cuda::subtract(mfloat,
  //                      cv::Scalar(sub_vals_[0], sub_vals_[1], sub_vals_[2]),
  //                      mfloat, cv::noArray(), -1);
  //   cv::cuda::divide(mfloat, cv::Scalar(div_vals_[0], div_vals_[1], div_vals_[2]),
  //                    mfloat, 1, -1);
  //   return mfloat;
  // }

  bool Model::doInference(
      const cv::Mat &img,
      std::unordered_map<std::string, std::vector<float>> &feature_vectors)
  {
    bool res = preProcess(img);

    std::cout << "runInference" << std::endl;
    auto succ = m_trtEngine->runInference(feature_vectors);
    if (!succ)
    {
      spdlog::error("Error: Unable to run inference.");
      return false;
    }
    return true;
  }

  // bool Model::doInference(
  //     cv::Mat &gpuImg,
  //     std::unordered_map<std::string, std::vector<float>> &feature_f_vectors,
  //       std::unordered_map<std::string, std::vector<int32_t>> &feature_int_vectors) {
  //   auto gpu_input = preprocess(gpuImg);
  //   auto succ = m_trtEngine->runInference(gpu_input, feature_f_vectors,feature_int_vectors);
  //   if (!succ) {
  //     spdlog::error("Error: Unable to run inference.");
  //     return false;
  //   }
  //   return true;
  // }
} // namespace tensorrt_inference