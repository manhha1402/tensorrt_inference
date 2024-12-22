#pragma once
#include "tensorrt_inference/detection.h"
namespace tensorrt_inference
{
    class YoloV8 : public Detection
    {
    public:
        // Builds the onnx model into a TensorRT engine, and loads the engine into
        // memory
        YoloV8(const std::string &model_name,
               tensorrt_inference::Options options = tensorrt_inference::Options(),
               const std::filesystem::path &model_dir =
                   std::filesystem::path(std::getenv("HOME")) / "data" / "weights");

    private:
        // Postprocess the output
        std::vector<Object> postprocess(
            std::unordered_map<std::string, std::vector<float>> &feature_vectors,
            const DetectionParams &params = DetectionParams(),
            const std::vector<std::string> &detected_class = {}) override;

        // Postprocess the output
        std::vector<Object> postprocessDetect(
            std::unordered_map<std::string, std::vector<float>> &feature_vectors,
            const DetectionParams &params = DetectionParams(),
            const std::vector<std::string> &detected_class = {});

        // Postprocess the output for pose model
        std::vector<Object> postprocessPose(
            std::unordered_map<std::string, std::vector<float>> &feature_vectors,
            const DetectionParams &params = DetectionParams(),
            const std::vector<std::string> &detected_class = {});

        // Postprocess the output for segmentation model
        std::vector<Object> postProcessSegmentation(
            std::unordered_map<std::string, std::vector<float>> &feature_vectors,
            const DetectionParams &params = DetectionParams(),
            const std::vector<std::string> &detected_class = {});
    };
} // namespace tensorrt_inference