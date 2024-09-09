#pragma once
#include "tensorrt_inference/face.h"
#include <fstream>
namespace tensorrt_inference
{
class RetinaFace : public Face
{
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    RetinaFace(const std::string& model_dir,const YAML::Node &config);

private:
    // Postprocess the output
    std::vector<FaceBox> postProcess(std::unordered_map<std::string, std::vector<float>> &feature_vectors) override;
    void create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h) override;
};
}