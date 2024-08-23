#include <tensorrt_inference/model.h>
namespace tensorrt_inference
{
Model::Model(const std::string& model_dir,const YAML::Node &config)
{
    onnx_file_ = model_dir + "/" + config["onnx_file"].as<std::string>();
    // INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    // IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    // IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    // image_order = config["image_order"].as<std::string>();
    // channel_order = config["channel_order"].as<std::string>();
    // img_mean = config["img_mean"].as<std::vector<float>>();
    // img_std = config["img_std"].as<std::vector<float>>();
    // alpha = config["alpha"].as<float>();
    // resize = config["resize"].as<std::string>();

    // Specify options for GPU inference
    Options options;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;
    options.engine_file_dir = getFolderOfFile(onnx_file_);


    // options.precision = config.precision;
    // options.calibrationDataDirectoryPath = config.calibrationDataDirectory;
    
    // if (options.precision == Precision::INT8) {
    //     if (options.calibrationDataDirectoryPath.empty()) {
    //         throw std::runtime_error("Error: Must supply calibration data path for INT8 calibration");
    //     }
    // }

    m_trtEngine = std::make_unique<Engine<float>>(options);
    auto succ = m_trtEngine->buildLoadNetwork(onnx_file_, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }

}
Model::~Model()
{

}
std::vector<std::vector<cv::cuda::GpuMat>> Model::preprocess(const cv::cuda::GpuMat &gpuImg)
{
    // Populate the input vectors
    const auto &inputDims = m_trtEngine->getInputDims();

    // Convert the image from BGR to RGB
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized = rgbMat;
    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

    // These params will be used in the post-processing stage
    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    return inputs;
}

bool Model::doInference(const cv::cuda::GpuMat &gpuImg,  std::vector<std::vector<std::vector<float>>>& feature_vectors)
{
    feature_vectors.clear();
    const auto input = preprocess(gpuImg);
    auto succ = m_trtEngine->runInference(input, feature_vectors);
    if (!succ) {
        std::string msg = "Error: Unable to run inference.";
        spdlog::error(msg);
        return false;
    }
    return true;
}

}