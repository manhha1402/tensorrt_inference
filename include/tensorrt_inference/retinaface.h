#pragma once
#include "tensorrt_inference/detection.h"
#include <fstream>
namespace tensorrt_inference
{
class RetinaFace : public Detection
{
struct anchorBox {
    float cx;
    float cy;
    float sx;
    float sy;
};
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    RetinaFace(const std::string& model_dir,const YAML::Node &config);

private:
    // Postprocess the output
    std::vector<Object> postprocess(std::unordered_map<std::string, std::vector<float>> &feature_vectors,const DetectionParams& params = DetectionParams()) override;

    void create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h) ;
};
}