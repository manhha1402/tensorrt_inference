#pragma once
#include "tensorrt_inference/detection.h"
namespace tensorrt_inference
{
class YoloV8 : public Detection 
{
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    YoloV8(const std::string& model_dir, const YAML::Node &config);

   
private:
    // Postprocess the output
    std::vector<Object> postprocess(std::unordered_map<std::string, std::vector<float>> &feature_vector) override;


   // Postprocess the output
    std::vector<Object> postprocessDetect(std::unordered_map<std::string, std::vector<float>> &feature_vector);

    // Postprocess the output for pose model
    std::vector<Object> postprocessPose(std::unordered_map<std::string, std::vector<float>> &feature_vector);

    // Postprocess the output for segmentation model
    std::vector<Object> postProcessSegmentation(std::unordered_map<std::string, std::vector<float>> &feature_vector);

};
}