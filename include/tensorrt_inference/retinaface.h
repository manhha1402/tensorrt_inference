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
    std::vector<FaceBox> postProcess(std::vector<std::vector<std::vector<float>>> &feature_vectors) override;
    void create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h) override;
    void nms(std::vector<FaceBox> &input_boxes, float NMS_THRESH);
    int anchor_num = 2;
    cv::Mat refer_matrix;
    std::vector<std::vector<int>> anchor_sizes;
};
}