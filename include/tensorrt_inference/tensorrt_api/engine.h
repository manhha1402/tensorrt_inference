#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "tensorrt_inference/tensorrt_api/logger.h"
#include "tensorrt_inference/tensorrt_api/util/Stopwatch.h"
#include "tensorrt_inference/tensorrt_api/util/Util.h"
#include <variant>

namespace tensorrt_inference
{
  // Precision used for GPU inference
  enum class Precision
  {
    // Full precision floating point value
    FP32,
    // Half prevision floating point value
    FP16,
    // Int8 quantization.
    // Has reduced dynamic range, may result in slight loss in accuracy.
    // If INT8 is selected, must provide path to calibration dataset directory.
    INT8,
  };

  // Options for the network
  struct Options
  {
    // Precision to use for GPU inference.
    Precision precision = Precision::FP16;
    // If INT8 precision is selected, must provide path to calibration dataset
    // directory.
    std::string calibrationDataDirectoryPath;
    // The batch size to be used when computing calibration data for INT8
    // inference. Should be set to as large a batch number as your GPU will
    // support.
    int32_t calibrationBatchSize = 128;
    // The batch size which should be optimized for.
    int32_t optBatchSize = 1;
    // Maximum allowable batch size
    int32_t maxBatchSize = 1;
    // GPU device index
    int deviceIndex = 0;
    // Directory where the engine file should be saved
    std::string engine_file_dir = "";
    // Maximum allowed input width
    int32_t maxInputWidth = -1; // Default to -1 --> expecting fixed input size
    // Minimum allowed input width
    int32_t minInputWidth = -1; // Default to -1 --> expecting fixed input size
    // Optimal input width
    int32_t optInputWidth = -1; // Default to -1 --> expecting fixed input size

    std::array<int, 4> MIN_DIMS_ = {1, 3, 20, 12};
    std::array<int, 4> OPT_DIMS_ = {1, 3, 256, 256};
    std::array<int, 4> MAX_DIMS_ = {1, 3, 960, 960};
    Options() {};
  };

  // Class to extend TensorRT logger
  class Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char *msg) noexcept override;
  };
  struct NetInfo
  {
    void *buffer;
    nvinfer1::Dims dims;
    size_t tensor_length = 0;
    nvinfer1::DataType data_type;
  };
  struct InferDeleter
  {
    template <typename T>
    void operator()(T *obj) const
    {
      delete obj;
    }
  };
  class Engine
  {
  public:
    Engine(const Options &options);
    ~Engine();

    // Build the onnx model into a TensorRT engine file, cache the model to disk
    // (to avoid rebuilding in future), and then load the model into memory.
    bool buildLoadNetwork(const std::string &onnx_file);

    // Load a TensorRT engine file from disk into memory
    // The default implementation will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f]
    // (some converted models may require this). If the model requires values to
    // be normalized between [-1.f, 1.f], use the following params:

    bool loadNetwork(const std::string engile_file);

    // Run inference.
    // Input format [input][batch][cv::cuda::GpuMat]
    // Output format [batch][output][feature_vector]
    bool runInference(
        std::unordered_map<std::string, std::vector<float>> &feature_vectors);
    bool runInference(
        float *input_buff,
        std::unordered_map<std::string, std::vector<float>> &feature_f_vectors,
        std::unordered_map<std::string, std::vector<int32_t>> &feature_int_vectors);

    // Holds pointers to the input and output GPU buffers
    std::unordered_map<std::string, NetInfo> input_map_;
    std::unordered_map<std::string, NetInfo> output_map_;

    // Utility method for resizing an image while maintaining the aspect ratio by
    // adding padding to smaller dimension after scaling While letterbox padding
    // normally adds padding to top & bottom, or left & right sides, this
    // implementation only adds padding to the right or bottom side This is done
    // so that it's easier to convert detected coordinates (ex. YOLO model) back
    // to the original reference frame.
    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(
        const cv::cuda::GpuMat &input, size_t height, size_t width,
        const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));

    [[nodiscard]] const std::unordered_map<std::string, NetInfo> &getInputInfo()
        const
    {
      return input_map_;
    };
    [[nodiscard]] const std::unordered_map<std::string, NetInfo> &getOutputInfo()
        const
    {
      return output_map_;
    };

    // Utility method for transforming triple nested output array into 2D array
    // Should be used when the output batch size is 1, but there are multiple
    // output feature vectors
    // static void transformOutput(std::vector<std::vector<std::vector<T>>> &input,
    //                             std::vector<std::vector<T>> &output);

    // // Utility method for transforming triple nested output array into single
    // // array Should be used when the output batch size is 1, and there is only a
    // // single output feature vector
    // static void transformOutput(std::vector<std::vector<std::vector<T>>> &input,
    //                             std::vector<T> &output);
    // Convert NHWC to NCHW and apply scaling and mean subtraction
    // static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat>
    // &batchInput, const std::array<float, 3> &subVals,
    //                                        const std::array<float, 3> &divVals,
    //                                        bool normalize, bool swapRB =
    //                                        false);
    virtual uint32_t getMaxOutputLength(const nvinfer1::Dims &tensorShape) const
    {
      uint32_t outputLength = 1;
      for (int j = 1; j < tensorShape.nbDims; ++j)
      {
        // We ignore j = 0 because that is the batch size, and we will take that
        // into account when sizing the buffer
        outputLength *= tensorShape.d[j];
      }
      return outputLength;
    }

  protected:
    const Options m_options;
    bool haveDynamicDims_;

    // Build the network
    bool build(const std::string &onnxModelPath);

    // Converts the engine options into a string
    std::string serializeEngineOptions(const Options &options,
                                       const std::string &onnxModelPath);

    void getDeviceNames(std::vector<std::string> &deviceNames);

    void clearGpuBuffers();

    // Must keep IRuntime around for inference, see:
    // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Logger m_logger;
  };

} // namespace tensorrt_inference
