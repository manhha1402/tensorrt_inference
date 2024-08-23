#pragma once
#include "tensorrt_inference/detection.h"
namespace tensorrt_inference
{
class YoloV9Refactor : public Detection 
{
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    YoloV9Refactor(const std::string& model_dir, const YAML::Node &config);

   
private:
    // Postprocess the output
    std::vector<Object> postprocessDetect(std::vector<float> &feature_vector) override;

};
}