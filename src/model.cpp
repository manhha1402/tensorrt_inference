#include <tensorrt_inference/model.h>
namespace tensorrt_inference
{
Model::Model(const std::string& model_dir,const YAML::Node &config)
{
    onnx_file_ = model_dir + "/" + config["onnx_file"].as<std::string>();
  
    // Specify options for GPU inference
    Options options;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;
    options.engine_file_dir = getFolderOfFile(onnx_file_);
    /*
    options.precision = config.precision;
    options.calibrationDataDirectoryPath = config.calibrationDataDirectory;
    
    if (options.precision == Precision::INT8) {
        if (options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: Must supply calibration data path for INT8 calibration");
        }
    }
    */
    m_trtEngine = std::make_unique<Engine<float>>(options);
    auto succ = m_trtEngine->buildLoadNetwork(onnx_file_);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }

}
Model::~Model()
{
}

cv::cuda::GpuMat Model::preprocess(const cv::cuda::GpuMat &gpuImg)
{
    // Populate the input vectors
    const auto & input_info = m_trtEngine->getInputInfo().begin();
    
   // These params will be used in the post-processing stage
    input_frame_h_ = gpuImg.rows;
    input_frame_w_ = gpuImg.cols;
    m_ratio = 1.f / std::min(input_info->second.dims.d[3] / static_cast<float>(gpuImg.cols), input_info->second.dims.d[2] / static_cast<float>(gpuImg.rows));
    // Convert the image from BGR to RGB
    cv::cuda::GpuMat rgbMat = gpuImg;
    auto resized = rgbMat;
    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (resized.rows != input_info->second.dims.d[2] || resized.cols != input_info->second.dims.d[3]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(rgbMat, input_info->second.dims.d[2], input_info->second.dims.d[3], cv::Scalar(128,128,128));
    }
    cv::cuda::GpuMat gpu_dst(1, resized.rows * resized.cols , CV_8UC3);
    size_t width = resized.cols * resized.rows;
    if (swapBR_) {
        std::vector<cv::cuda::GpuMat> input_channels{
            cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U, &(gpu_dst.ptr()[width * 2 ])),
            cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U, &(gpu_dst.ptr()[width])),
            cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U, &(gpu_dst.ptr()[0 ]))};
        cv::cuda::split(resized, input_channels); // HWC -> CHW
    } else {
        std::vector<cv::cuda::GpuMat> input_channels{
            cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U, &(gpu_dst.ptr()[0 ])),
            cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U, &(gpu_dst.ptr()[width ])),
            cv::cuda::GpuMat(resized.rows, resized.cols, CV_8U, &(gpu_dst.ptr()[width * 2]))};
        cv::cuda::split(resized, input_channels); // HWC -> CHW
    }
    cv::cuda::GpuMat mfloat;
    if (normalized_) {
        // [0.f, 1.f]
        gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
        for(auto& val : sub_vals_)
            val = val/255.0f;
    } else {
        // [0.f, 255.f]
        gpu_dst.convertTo(mfloat, CV_32FC3);
    }
    
    // Apply scaling and mean subtraction
    cv::cuda::subtract(mfloat, cv::Scalar(sub_vals_[0], sub_vals_[1], sub_vals_[2]), mfloat, cv::noArray(), -1);
    cv::cuda::divide(mfloat, cv::Scalar(div_vals_[0], div_vals_[1], div_vals_[2]), mfloat, 1, -1);
    return mfloat;
   
}
 
bool Model::doInference(cv::cuda::GpuMat &gpuImg,  std::unordered_map<std::string, std::vector<float>>& feature_vectors)
{
    auto gpu_input = preprocess(gpuImg);
    auto succ = m_trtEngine->runInference(gpu_input, feature_vectors);
    if (!succ) {
        std::string msg = "Error: Unable to run inference.";
        spdlog::error(msg);
        return false;
    }
    return true;
}

}