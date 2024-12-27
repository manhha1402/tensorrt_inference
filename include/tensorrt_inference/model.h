
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

        void setParams(const PreprocessParams &params);

    protected:
        /**
         * Preprocess image beforing doing the inference
         */
        bool preProcess(const cv::Mat &img);

        void setDefaultParams(const std::string &model_name);
        std::string onnx_file_;
        std::vector<float> factors_;

        PreprocessParams preprocess_params_;

        int num_kps_ = 17;
        std::vector<float> ratios_{1, 1};
        float input_frame_w_, input_frame_h_;
    };
} // namespace tensorrt_inference
