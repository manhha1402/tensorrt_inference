#include "tensorrt_inference/tensorrt_api/engine.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
namespace tensorrt_inference {
using namespace nvinfer1;
using namespace Util;

void Logger::log(Severity severity, const char *msg) noexcept {
  switch (severity) {
  case Severity::kVERBOSE:
    spdlog::debug(msg);
    break;
  case Severity::kINFO:
    spdlog::info(msg);
    break;
  case Severity::kWARNING:
    spdlog::warn(msg);
    break;
  case Severity::kERROR:
    spdlog::error(msg);
    break;
  case Severity::kINTERNAL_ERROR:
    spdlog::critical(msg);
    break;
  default:
    spdlog::info("Unexpected severity level");
  }
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(
    int32_t batchSize, int32_t inputW, int32_t inputH,
    const std::string &calibDataDirPath, const std::string &calibTableName,
    const std::string &inputBlobName, const std::array<float, 3> &subVals,
    const std::array<float, 3> &divVals, bool normalize, bool readCache)
    : m_batchSize(batchSize), m_inputW(inputW), m_inputH(inputH), m_imgIdx(0),
      m_calibTableName(calibTableName), m_inputBlobName(inputBlobName),
      m_subVals(subVals), m_divVals(divVals), m_normalize(normalize),
      m_readCache(readCache) {

  // Allocate GPU memory to hold the entire batch
  m_inputCount = 3 * inputW * inputH * batchSize;
  checkCudaErrorCode(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));

  // Read the name of all the files in the specified directory.
  if (!doesFileExist(calibDataDirPath)) {
    auto msg =
        "Error, directory at provided path does not exist: " + calibDataDirPath;
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }

  m_imgPaths = getFilesInDirectory(calibDataDirPath);
  if (m_imgPaths.size() < static_cast<size_t>(batchSize)) {
    auto msg = "Error, there are fewer calibration images than the specified "
               "batch size!";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }

  // Randomize the calibration data
  auto rd = std::random_device{};
  auto rng = std::default_random_engine{rd()};
  std::shuffle(std::begin(m_imgPaths), std::end(m_imgPaths), rng);
}

int32_t Int8EntropyCalibrator2::getBatchSize() const noexcept {
  // Return the batch size
  return m_batchSize;
}

bool Int8EntropyCalibrator2::getBatch(void **bindings, const char **names,
                                      int32_t nbBindings) noexcept {
  // // This method will read a batch of images into GPU memory, and place the
  // // pointer to the GPU memory in the bindings variable.

  // if (m_imgIdx + m_batchSize > static_cast<int>(m_imgPaths.size())) {
  //     // There are not enough images left to satisfy an entire batch
  //     return false;
  // }

  // // Read the calibration images into memory for the current batch
  // std::vector<cv::cuda::GpuMat> inputImgs;
  // for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++) {
  //     spdlog::info("Reading image {}: {}", i, m_imgPaths[i]);
  //     auto cpuImg = cv::imread(m_imgPaths[i]);
  //     if (cpuImg.empty()) {
  //         spdlog::error("Fatal error: Unable to read image at path: " +
  //         m_imgPaths[i]); return false;
  //     }

  //     cv::cuda::GpuMat gpuImg;
  //     gpuImg.upload(cpuImg);
  //     //cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

  //     // TODO: Define any preprocessing code here, such as resizing
  //     auto resized =
  //     Engine<float>::resizeKeepAspectRatioPadRightBottom(gpuImg, m_inputH,
  //     m_inputW);

  //     inputImgs.emplace_back(std::move(resized));
  // }

  // // Convert the batch from NHWC to NCHW
  // // ALso apply normalization, scaling, and mean subtraction
  // auto mfloat = Engine<float>::blobFromGpuMats(inputImgs, m_subVals,
  // m_divVals, m_normalize, true); auto *dataPointer = mfloat.ptr<void>();

  // // Copy the GPU buffer to member variable so that it persists
  // checkCudaErrorCode(cudaMemcpyAsync(m_deviceInput, dataPointer, m_inputCount
  // * sizeof(float), cudaMemcpyDeviceToDevice));

  // m_imgIdx += m_batchSize;
  // if (std::string(names[0]) != m_inputBlobName) {
  //     spdlog::error("Error: Incorrect input name provided!");
  //     return false;
  // }
  // bindings[0] = m_deviceInput;
  return true;
}

void const *
Int8EntropyCalibrator2::readCalibrationCache(size_t &length) noexcept {
  spdlog::info("Searching for calibration cache: {}", m_calibTableName);
  m_calibCache.clear();
  std::ifstream input(m_calibTableName, std::ios::binary);
  input >> std::noskipws;
  if (m_readCache && input.good()) {
    spdlog::info("Reading calibration cache: {}", m_calibTableName);
    std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
              std::back_inserter(m_calibCache));
  }
  length = m_calibCache.size();
  return length ? m_calibCache.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(
    const void *ptr, std::size_t length) noexcept {
  spdlog::info("Writing calibration cache: {}", m_calibTableName);
  spdlog::info("Calibration cache size: {} bytes", length);
  std::ofstream output(m_calibTableName, std::ios::binary);
  output.write(reinterpret_cast<const char *>(ptr), length);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() {
  checkCudaErrorCode(cudaFree(m_deviceInput));
};
/////////////////////////////////////////////////////////////////

Engine::Engine(const Options &options) : m_options(options) {}

Engine::~Engine() {
  clearGpuBuffers();
}
bool Engine::buildLoadNetwork(const std::string& onnx_file) {
  // get engine name
  const auto engine_name = serializeEngineOptions(m_options, onnx_file);
  // get engine directory
  const auto engine_dir = std::filesystem::path(m_options.engine_file_dir);
  std::filesystem::path engine_path = engine_dir / engine_name;
  spdlog::info("Searching for engine file with name: {}", engine_path.string());

  if (Util::doesFileExist(engine_path)) {
    spdlog::info("Engine found, not regenerating...");
  } else {
    if (!Util::doesFileExist(onnx_file)) {
      auto msg = "Could not find ONNX model at path: " + onnx_file;
      spdlog::error(msg);
      throw std::runtime_error(msg);
    }

    spdlog::info("Engine not found, generating. This could take a while...");
    if (!std::filesystem::exists(engine_dir)) {
      std::filesystem::create_directories(engine_dir);
      spdlog::info("Created directory: {}", engine_dir.string());
    }
    // init plugin 
    initLibNvInferPlugins(&m_logger, "");
    auto ret = build(onnx_file);
    if (!ret) {
      return false;
    }
  }

  return loadNetwork(engine_path);
}

bool Engine::loadNetwork(std::string trtModelPath) {
  // Read the serialized model from disk
  if (!Util::doesFileExist(trtModelPath)) {
    auto msg = "Error, unable to read TensorRT model at path: " + trtModelPath;
    spdlog::error(msg);
    return false;
  } else {
    auto msg = "Loading TensorRT engine file at path: " + trtModelPath;
    spdlog::info(msg);
  }

  std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    auto msg = "Error, unable to read engine file";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }
  initLibNvInferPlugins(&m_logger, "");
  // Create a runtime to deserialize the engine file.
  m_runtime = std::unique_ptr<nvinfer1::IRuntime>{
      nvinfer1::createInferRuntime(m_logger)};
  if (!m_runtime) {
    return false;
  }
  // Set the device index
  auto ret = cudaSetDevice(m_options.deviceIndex);
  if (ret != 0) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    auto errMsg = "Unable to set GPU device index to: " +
                  std::to_string(m_options.deviceIndex) +
                  ". Note, your device has " + std::to_string(numGPUs) +
                  " CUDA-capable GPU(s).";
    spdlog::error(errMsg);
    throw std::runtime_error(errMsg);
  }

  // Create an engine, a representation of the optimized model.
  m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!m_engine) {
    return false;
  }

  // The execution context contains all of the state associated with a
  // particular invocation
  m_context = std::unique_ptr<nvinfer1::IExecutionContext>(
      m_engine->createExecutionContext());
  if (!m_context) {
    return false;
  }

  // Storage for holding the input and output buffers
  // This will be passed to TensorRT for inference
  clearGpuBuffers();
  input_map_.clear();
  output_map_.clear();
  // Create a cuda stream
  cudaStream_t stream;
  Util::checkCudaErrorCode(cudaStreamCreate(&stream));
  spdlog::info("Allocate GPU memory for input and output buffers");

  // Allocate GPU memory for input and output buffers
  for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
    const auto tensorName = m_engine->getIOTensorName(i);
    const auto tensorType = m_engine->getTensorIOMode(tensorName);
    const auto tensorShape =
        m_engine->getTensorShape(tensorName);  // getBindingDimensions
    const auto tensorDataType = m_engine->getTensorDataType(tensorName);
    if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
      // Store the input dims for later use
      input_map_[tensorName].dims = tensorShape;
      input_map_[tensorName].data_type = tensorDataType;
      // input_map_[tensorName].tensor_length = tensor_length;
      // Util::checkCudaErrorCode(cudaMallocAsync(
      //     &input_map_[tensorName].buffer, tensor_length * sizeof(T),
      //     stream));

    } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
      
      // The binding is an output
      uint32_t tensor_length = 1;
      haveDynamicDims_ = tensorShape.d[1] == -1 || tensorShape.d[2] == -1 ||
                         tensorShape.d[3] == -1;
      if (haveDynamicDims_) {
        tensor_length = getMaxOutputLength(tensorShape);
      } else {
        for (int j = 1; j < tensorShape.nbDims; ++j) {
          // We ignore j = 0 because that is the batch size, and we will take
          // that into account when sizing the buffer
          tensor_length *= tensorShape.d[j];
        }
      }
      // Now size the output buffer appropriately, taking into account the max
      // possible batch size (although we could actually end up using less
      // memory)
      output_map_[tensorName].dims = tensorShape;
      output_map_[tensorName].tensor_length = tensor_length;
      output_map_[tensorName].data_type = tensorDataType;
      //TODO not sure
      Util::checkCudaErrorCode(cudaMallocAsync(
          &output_map_[tensorName].buffer, tensor_length * sizeof(tensorDataType), stream));

    } else {
      auto msg = "Error, IO Tensor is neither an input or output!";
      spdlog::error(msg);
      throw std::runtime_error(msg);
    }
  }
  spdlog::info("In/Out dimensions of model: ({}, {})", input_map_.size(),
               output_map_.size());
  for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
    spdlog::info("Name {} : Output dimensions: ({}, {}, {}, {})", it->first,
                 it->second.dims.d[0], it->second.dims.d[1],
                 it->second.dims.d[2], it->second.dims.d[3]);
  }
  for (auto it = input_map_.begin(); it != input_map_.end(); ++it) {
    spdlog::info("Name {} : Input dimensions: ({}, {}, {}, {})", it->first,
                 it->second.dims.d[0], it->second.dims.d[1],
                 it->second.dims.d[2], it->second.dims.d[3]);
  }
  // Synchronize and destroy the cuda stream
  Util::checkCudaErrorCode(cudaStreamSynchronize(stream));
  Util::checkCudaErrorCode(cudaStreamDestroy(stream));

  return true;
}

bool Engine::build(const std::string& onnxModelPath) {
  // Create our engine builder.
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(m_logger));
  if (!builder) {
    return false;
  }

  // Define an explicit batch size and then create the network (implicit batch
  // size is deprecated). More info here:
  // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
  auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));
  if (!network) {
    return false;
  }

  // Create a parser for reading the onnx file.
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, m_logger));
  if (!parser) {
    return false;
  }

  // We are going to first read the onnx file into memory, then pass that buffer
  // to the parser. Had our onnx model file been encrypted, this approach would
  // allow us to first decrypt the buffer.
  std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    auto msg = "Error, unable to read engine file";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }

  // Parse the buffer we read into memory.
  auto parsed = parser->parse(buffer.data(), buffer.size());
  if (!parsed) {
    return false;
  }

  // Ensure that all the inputs have the same batch size
  const auto numInputs = network->getNbInputs();
  if (numInputs < 1) {
    auto msg = "Error, model needs at least 1 input!";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }
  const auto input0Batch = network->getInput(0)->getDimensions().d[0];
  for (int32_t i = 1; i < numInputs; ++i) {
    if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
      auto msg =
          "Error, the model has multiple inputs, each with differing batch "
          "sizes!";
      spdlog::error(msg);
      throw std::runtime_error(msg);
    }
  }

  // Check to see if the model supports dynamic batch size or not
  bool doesSupportDynamicBatch = false;
  if (input0Batch == -1) {
    doesSupportDynamicBatch = true;
    spdlog::info("Model supports dynamic batch size");
  } else {
    spdlog::info("Model only supports fixed batch size of {}", input0Batch);
    // If the model supports a fixed batch size, ensure that the maxBatchSize
    // and optBatchSize were set correctly.
    if (m_options.optBatchSize != input0Batch ||
        m_options.maxBatchSize != input0Batch) {
      auto msg =
          "Error, model only supports a fixed batch size of " +
          std::to_string(input0Batch) +
          ". Must set Options.optBatchSize and Options.maxBatchSize to 1";
      spdlog::error(msg);
      throw std::runtime_error(msg);
    }
  }
  const auto input3Batch = network->getInput(0)->getDimensions().d[3];
  //   bool doesSupportDynamicWidth = false;
  //   if (input3Batch == -1) {
  //     doesSupportDynamicWidth = true;
  //     spdlog::info(
  //         "Model supports dynamic width. Using Options.maxInputWidth, "
  //         "Options.minInputWidth, and Options.optInputWidth to set the input
  //         " "width.");

  //     // Check that the values of maxInputWidth, minInputWidth, and
  //     optInputWidth
  //     // are valid

  //     if (m_options.maxInputWidth < m_options.minInputWidth ||
  //         m_options.maxInputWidth < m_options.optInputWidth ||
  //         m_options.minInputWidth > m_options.optInputWidth ||
  //         m_options.maxInputWidth < 1 || m_options.minInputWidth < 1 ||
  //         m_options.optInputWidth < 1) {
  //       auto msg =
  //           "Error, invalid values for Options.maxInputWidth, "
  //           "Options.minInputWidth, and Options.optInputWidth";
  //       spdlog::error(msg);
  //       throw std::runtime_error(msg);
  //     }
  //   }

  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }

  // Register a single optimization profile
  nvinfer1::IOptimizationProfile* optProfile =
      builder->createOptimizationProfile();
  for (int32_t i = 0; i < numInputs; ++i) {
    // Must specify dimensions for all the inputs the model expects.
    const auto input = network->getInput(i);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    int32_t inputC = inputDims.d[1];
    int32_t inputH = inputDims.d[2];
    int32_t inputW = inputDims.d[3];
    int32_t minInputWidth = std::max(m_options.minInputWidth, inputW);
    spdlog::info("Input name and dimensions of model: ({} : {}, {}, {})",
                 inputName, inputC, inputH, inputW);

    haveDynamicDims_ = inputC == -1 || inputH == -1 || inputW == -1;
    // Specify the optimization profile`
    if (haveDynamicDims_) {
       spdlog::info("Model has dynamic range, set manual dimensions");
      optProfile->setDimensions(
          network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN,
          nvinfer1::Dims4(m_options.MIN_DIMS_[0], m_options.MIN_DIMS_[1],
                          m_options.MIN_DIMS_[2], m_options.MIN_DIMS_[3]));
      optProfile->setDimensions(
          network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT,
          nvinfer1::Dims4(m_options.OPT_DIMS_[0], m_options.OPT_DIMS_[1],
                          m_options.OPT_DIMS_[2], m_options.OPT_DIMS_[3]));
      optProfile->setDimensions(
          network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX,
          nvinfer1::Dims4(m_options.MAX_DIMS_[0], m_options.MAX_DIMS_[1],
                          m_options.MAX_DIMS_[2], m_options.MAX_DIMS_[3]));
    } else {
      optProfile->setDimensions(
          inputName, nvinfer1::OptProfileSelector::kMIN,
          nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
      optProfile->setDimensions(
          inputName, nvinfer1::OptProfileSelector::kOPT,
          nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
      optProfile->setDimensions(
          inputName, nvinfer1::OptProfileSelector::kMAX,
          nvinfer1::Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    }
  }
  config->addOptimizationProfile(optProfile);
  // Set the precision level
  const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
  if (m_options.precision == Precision::FP16) {
    // Ensure the GPU supports FP16 inference
    if (!builder->platformHasFastFp16()) {
      auto msg = "Error: GPU does not support FP16 precision";
      spdlog::error(msg);
      throw std::runtime_error(msg);
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (m_options.precision == Precision::INT8) {
    if (numInputs > 1) {
      auto msg =
          "Error, this implementation currently only supports INT8 "
          "quantization for single input models";
      spdlog::error(msg);
      throw std::runtime_error(msg);
    }

    // Ensure the GPU supports INT8 Quantization
    if (!builder->platformHasFastInt8()) {
      auto msg = "Error: GPU does not support INT8 precision";
      spdlog::error(msg);
      throw std::runtime_error(msg);
    }

    // Ensure the user has provided path to calibration data directory
    if (m_options.calibrationDataDirectoryPath.empty()) {
      auto msg =
          "Error: If INT8 precision is selected, must provide path to "
          "calibration data directory to Engine::build method";
      throw std::runtime_error(msg);
    }
    config->setFlag((nvinfer1::BuilderFlag::kINT8));

    const auto input = network->getInput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    const auto calibrationFileName = engineName + ".calibration";

    m_calibrator = std::make_unique<Int8EntropyCalibrator2>(
        m_options.calibrationBatchSize, inputDims.d[3], inputDims.d[2],
        m_options.calibrationDataDirectoryPath, calibrationFileName, inputName);
    config->setInt8Calibrator(m_calibrator.get());
  }

  // CUDA stream used for profiling by the builder.
  cudaStream_t profileStream;
  Util::checkCudaErrorCode(cudaStreamCreate(&profileStream));
  config->setProfileStream(profileStream);

  // Build the engine
  // If this call fails, it is suggested to increase the logger verbosity to
  // kVERBOSE and try rebuilding the engine. Doing so will provide you with more
  // information on why exactly it is failing.
  std::unique_ptr<nvinfer1::IHostMemory> plan{
      builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    return false;
  }

  // Write the engine to disk
  const auto enginePath =
      std::filesystem::path(m_options.engine_file_dir) / engineName;
  std::ofstream outfile(enginePath, std::ofstream::binary);
  outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
  spdlog::info("Success, saved engine to {}", enginePath.string());
  Util::checkCudaErrorCode(cudaStreamDestroy(profileStream));
  return true;
}

bool Engine::runInference(
    cv::cuda::GpuMat &input,
    std::unordered_map<std::string, std::vector<float>> &feature_vectors) {
  if (input.empty()) {
    spdlog::error("Provided input vector is empty!");
    return false;
  }
  // Create the cuda stream that will be used for inference
  cudaStream_t inferenceCudaStream;
  Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
  input_map_[input_map_.begin()->first].tensor_length =
      input.cols * input.rows * input.channels();

  // Set the address of the input buffers
  bool status = m_context->setTensorAddress(input_map_.begin()->first.c_str(),
                                            input.ptr<void>());
  if (!status) {
    return false;
  }
  for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
    bool status =
        m_context->setTensorAddress(it->first.c_str(), it->second.buffer);
    if (!status) {
      return false;
    }
  }
  if (!m_context->allInputDimensionsSpecified()) {
    auto msg = "Error, not all required dimensions specified.";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }
  // Run inference.
  status = m_context->enqueueV3(inferenceCudaStream);
  if (!status) {
    return false;
  }
  // Copy the outputs back to CPU
  for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
    if(it->second.data_type != nvinfer1::DataType::kFLOAT )
    {
      auto msg = "Output must be float type";
      spdlog::error(msg);
      throw std::runtime_error(msg);
    }
    feature_vectors[it->first].resize(it->second.tensor_length);
    // Copy the output
    Util::checkCudaErrorCode(
        cudaMemcpyAsync(feature_vectors[it->first].data(), it->second.buffer,
                        it->second.tensor_length * sizeof(it->second.data_type),
                        cudaMemcpyDeviceToHost, inferenceCudaStream));
  }

  // Synchronize the cuda stream
  Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
  Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
  return true;
}
bool Engine::runInference(
      cv::cuda::GpuMat &input,
      std::unordered_map<std::string, std::vector<float>> &feature_f_vectors,
      std::unordered_map<std::string, std::vector<int32_t>> &feature_int_vectors)
  {

    if (input.empty()) {
      spdlog::error("Provided input vector is empty!");
      return false;
    }
    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
    input_map_[input_map_.begin()->first].tensor_length =
        input.cols * input.rows * input.channels();

    // Set the address of the input buffers
    bool status = m_context->setTensorAddress(input_map_.begin()->first.c_str(),
                                              input.ptr<void>());
    if (!status) {
      return false;
    }
    for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
      bool status =
          m_context->setTensorAddress(it->first.c_str(), it->second.buffer);
      if (!status) {
        return false;
      }
    }
    if (!m_context->allInputDimensionsSpecified()) {
      auto msg = "Error, not all required dimensions specified.";
      spdlog::error(msg);
      throw std::runtime_error(msg);
    }
    // Run inference.
    status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
      return false;
    }
    // Copy the outputs back to CPU
    for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
      if(it->second.data_type == nvinfer1::DataType::kFLOAT )
      {
        feature_f_vectors[it->first].resize(it->second.tensor_length);
        // Copy the output
        Util::checkCudaErrorCode(
            cudaMemcpyAsync(feature_f_vectors[it->first].data(), it->second.buffer,
                            it->second.tensor_length * sizeof(it->second.data_type),
                            cudaMemcpyDeviceToHost, inferenceCudaStream));  
      }
      else if(it->second.data_type == nvinfer1::DataType::kINT32)
      {
        feature_int_vectors[it->first].resize(it->second.tensor_length);
        // Copy the output
        Util::checkCudaErrorCode(
            cudaMemcpyAsync(feature_int_vectors[it->first].data(), it->second.buffer,
                            it->second.tensor_length * sizeof(it->second.data_type),
                            cudaMemcpyDeviceToHost, inferenceCudaStream));
      }
      else{
        auto msg = "Unknown output type";
        spdlog::error(msg);
        throw std::runtime_error(msg);
      }
    }

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
  }




cv::cuda::GpuMat Engine::resizeKeepAspectRatioPadRightBottom(
    const cv::cuda::GpuMat &input, size_t height, size_t width,
    const cv::Scalar &bgcolor) {
  float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
  int unpad_w = r * input.cols;
  int unpad_h = r * input.rows;
  cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
  cv::cuda::resize(input, re, re.size());
  cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
  re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
  return out;
}

void Engine::getDeviceNames(std::vector<std::string> &deviceNames) {
  int numGPUs;
  cudaGetDeviceCount(&numGPUs);

  for (int device = 0; device < numGPUs; device++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    deviceNames.push_back(std::string(prop.name));
  }
}

std::string Engine::serializeEngineOptions(
    const Options &options, const std::string &onnxModelPath) {
  const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
  std::string engineName =
      onnxModelPath.substr(filenamePos,
                           onnxModelPath.find_last_of('.') - filenamePos) +
      ".engine";

  // Add the GPU device name to the file to ensure that the model is only used
  // on devices with the exact same GPU
  std::vector<std::string> deviceNames;
  getDeviceNames(deviceNames);

  if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
    auto msg = "Error, provided device index is out of range!";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }

  auto deviceName = deviceNames[options.deviceIndex];
  // Remove spaces from the device name
  deviceName.erase(
      std::remove_if(deviceName.begin(), deviceName.end(), ::isspace),
      deviceName.end());

  engineName += "." + deviceName;

  // Serialize the specified options into the filename
  if (options.precision == Precision::FP16) {
    engineName += ".fp16";
  } else if (options.precision == Precision::FP32) {
    engineName += ".fp32";
  } else {
    engineName += ".int8";
  }

  engineName += "." + std::to_string(options.maxBatchSize);
  engineName += "." + std::to_string(options.optBatchSize);
  engineName += "." + std::to_string(options.minInputWidth);
  engineName += "." + std::to_string(options.optInputWidth);
  engineName += "." + std::to_string(options.maxInputWidth);
  spdlog::info("Engine name: {}", engineName);
  return engineName;
}

void Engine::clearGpuBuffers() {
  if (!input_map_.empty()) {
    // Free GPU memory of inputs
    for (auto it = input_map_.begin(); it != input_map_.end(); ++it) {
      Util::checkCudaErrorCode(cudaFree(it->second.buffer));
    }
    input_map_.clear();
  }
  if (!output_map_.empty()) {
    // Free GPU memory of inputs
    for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
      Util::checkCudaErrorCode(cudaFree(it->second.buffer));
    }
    output_map_.clear();
  }
}
} // namespace 