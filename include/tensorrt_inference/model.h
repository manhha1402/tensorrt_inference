
#pragma once

#include <yaml-cpp/yaml.h>
#include <variant>
#include <opencv2/opencv.hpp>

#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"
namespace tensorrt_inference
{

    class Model
    {
    public:
        explicit Model(
            const std::string &model_name,
            tensorrt_inference::Options options = tensorrt_inference::Options(),
            const std::filesystem::path &model_dir =
                std::filesystem::path(std::getenv("HOME")) / "data" / "weights");
        ~Model();
        bool doInference(
            const cv::Mat &img,
            std::unordered_map<std::string, std::vector<float>> &feature_vectors);
        // bool doInference(
        //   cv::Mat &img,
        //   std::unordered_map<std::string, std::vector<float>> &feature_f_vectors,
        //     std::unordered_map<std::string, std::vector<int32_t>> &feature_int_vectors);
        std::unique_ptr<Engine> m_trtEngine = nullptr;

    protected:
        // Preprocess the input. Normalize values between [0.f, 1.f] Setting the
        // normalize flag to false will leave values between [0.f, 255.f] (some
        // converted models may require this). If the model requires values to be
        // normalized between [-1.f, 1.f], use the following params:
        //    subVals = {0.0f, 0.0f, 0.0f};
        //    divVals = {1.f, 1.f, 1.f};
        //    normalize = false;
        // cv::cuda::GpuMat preprocess(const cv::cuda::GpuMat &gpuImg);

        bool preProcess(const cv::Mat &img);

        std::string onnx_file_;
        std::vector<float> factors_;

        std::vector<float> sub_vals_{0, 0, 0};
        std::vector<float> div_vals_{1.0f, 1.0f, 1.0f};
        bool normalized_ = false;
        bool swapBR_ = true;
        int num_kps_ = 17;
        bool keep_ratio_ = true;
        std::vector<float> ratios_{1, 1};
        float input_frame_w_, input_frame_h_;
    };
} // namespace tensorrt_inference
