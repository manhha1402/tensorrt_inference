#pragma once
#include "tensorrt_inference/detection.h"
namespace tensorrt_inference
{
class YoloV9 : public Detection 
{
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    YoloV9(const std::string& model_dir, const YAML::Node &config);

   
private:
    // Postprocess the output
    std::vector<Object> postprocessDetect(std::unordered_map<std::string, std::vector<float>> &feature_vector) override;

};
}