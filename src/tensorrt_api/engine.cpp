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

namespace {
  int toSizeOf(const nvinfer1::DataType& data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kINT32:
            return sizeof(int32_t);
        case nvinfer1::DataType::kINT8:
            return sizeof(int8_t);
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kBOOL:
            return sizeof(bool);
        default:
            spdlog::error("Unknown data type: {}", static_cast<int>(data_type));
            throw std::runtime_error("Unknown data type encountered in toSizeOf");
    }
}
}



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
    uint32_t tensor_length = 1;
    for (int j = 1; j < tensorShape.nbDims; ++j) {
          // We ignore j = 0 because that is the batch size, and we will take
          // that into account when sizing the buffer
          tensor_length *= tensorShape.d[j];
        }
    if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
      // Store the input dims for later use
      input_map_[tensorName].dims = tensorShape;
      input_map_[tensorName].data_type = tensorDataType;
      input_map_[tensorName].tensor_length = tensor_length;
      Util::checkCudaErrorCode(cudaMallocAsync(
          &input_map_[tensorName].buffer, tensor_length * toSizeOf(tensorDataType),
          stream));

    } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
      // The binding is an output
      
      haveDynamicDims_ = tensorShape.d[1] == -1 || tensorShape.d[2] == -1 ||
                         tensorShape.d[3] == -1;
      if (haveDynamicDims_) {
        tensor_length = getMaxOutputLength(tensorShape);
      }
      // Now size the output buffer appropriately, taking into account the max
      // possible batch size (although we could actually end up using less
      // memory)
      output_map_[tensorName].dims = tensorShape;
      output_map_[tensorName].tensor_length = tensor_length;
      output_map_[tensorName].data_type = tensorDataType;
      Util::checkCudaErrorCode(cudaMallocAsync(
          &output_map_[tensorName].buffer, tensor_length * toSizeOf(tensorDataType), stream));

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
  // Ensure the GPU supports FP16 inference
  if (!builder->platformHasFastFp16()) {
    auto msg = "Error: GPU does not support FP16 precision";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }
  config->setFlag(nvinfer1::BuilderFlag::kTF32);


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
    std::unordered_map<std::string, std::vector<float>> &feature_vectors) {
  // Create the cuda stream that will be used for inference
  cudaStream_t inferenceCudaStream;
  Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
  // Set the address of the input buffers and copy cpu to gpu buffer
  // Copy the output
  // Util::checkCudaErrorCode(
  //       cudaMemcpyAsync(input_map_.begin()->second.buffer, input_buff,
  //                        input_map_.begin()->second.tensor_length,
  //                       cudaMemcpyHostToDevice, inferenceCudaStream));
  
  bool status = m_context->setTensorAddress(input_map_.begin()->first.c_str(),
                                            input_map_.begin()->second.buffer);
  if (!status) {
    return false;
  }
   // Set the address of the output buffers 
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
                        it->second.tensor_length * toSizeOf(it->second.data_type),
                        cudaMemcpyDeviceToHost, inferenceCudaStream));
  }
 
  // Synchronize the cuda stream
  Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
  Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
  return true;
}
// bool Engine::runInference(
//       float* input_buff,
//       std::unordered_map<std::string, std::vector<float>> &feature_f_vectors,
//       std::unordered_map<std::string, std::vector<int32_t>> &feature_int_vectors)
//   {

//     if (input.empty()) {
//       spdlog::error("Provided input vector is empty!");
//       return false;
//     }
//     // Create the cuda stream that will be used for inference
//     cudaStream_t inferenceCudaStream;
//     Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
//     input_map_[input_map_.begin()->first].tensor_length =
//         input.cols * input.rows * input.channels();

//     // Set the address of the input buffers
//     bool status = m_context->setTensorAddress(input_map_.begin()->first.c_str(),
//                                               input.ptr<void>());
//     if (!status) {
//       return false;
//     }
//     for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
//       bool status =
//           m_context->setTensorAddress(it->first.c_str(), it->second.buffer);
//       if (!status) {
//         return false;
//       }
//     }
//     if (!m_context->allInputDimensionsSpecified()) {
//       auto msg = "Error, not all required dimensions specified.";
//       spdlog::error(msg);
//       throw std::runtime_error(msg);
//     }
//     // Run inference.
//     status = m_context->enqueueV3(inferenceCudaStream);
//     if (!status) {
//       return false;
//     }
//     // Copy the outputs back to CPU
//     for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
//       if(it->second.data_type == nvinfer1::DataType::kFLOAT )
//       {
//         feature_f_vectors[it->first].resize(it->second.tensor_length);
//         // Copy the output
//         Util::checkCudaErrorCode(
//             cudaMemcpyAsync(feature_f_vectors[it->first].data(), it->second.buffer,
//                             it->second.tensor_length * toSizeOf(it->second.data_type),
//                             cudaMemcpyDeviceToHost, inferenceCudaStream));  
//       }
//       else if(it->second.data_type == nvinfer1::DataType::kINT32)
//       {
//         feature_int_vectors[it->first].resize(it->second.tensor_length);
//         // Copy the output
//         Util::checkCudaErrorCode(
//             cudaMemcpyAsync(feature_int_vectors[it->first].data(), it->second.buffer,
//                             it->second.tensor_length * toSizeOf(it->second.data_type),
//                             cudaMemcpyDeviceToHost, inferenceCudaStream));
//       }
//       else{
//         auto msg = "Unknown output type";
//         spdlog::error(msg);
//         throw std::runtime_error(msg);
//       }
//     }

//     // Synchronize the cuda stream
//     Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
//     Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
//     return true;
//   }




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