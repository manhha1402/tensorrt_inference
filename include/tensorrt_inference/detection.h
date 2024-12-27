#pragma once
#include <tensorrt_inference/model.h>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "tensorrt_inference/tensorrt_api/engine.h"
#include "tensorrt_inference/utils.h"

namespace tensorrt_inference
{

    class Detection : public Model
    {
    public:
        explicit Detection(
            const std::string &model_name,
            tensorrt_inference::Options options = tensorrt_inference::Options(),
            const std::filesystem::path &model_dir =
                std::filesystem::path(std::getenv("HOME")) / "data" / "weights");

        // Detect the objects in the image
        std::vector<Object> detect(
            const cv::Mat &inputImageBGR,
            const DetectionParams &params = DetectionParams(),
            const std::vector<std::string> &detected_class = {});
        std::vector<Object> detect(
            cv::cuda::GpuMat &inputImageBGR,
            const DetectionParams &params = DetectionParams(),
            const std::vector<std::string> &detected_class = {});
        cv::Mat drawObjectLabels(const cv::Mat &image,
                                 const std::vector<Object> &objects,
                                 const DetectionParams &params = DetectionParams(),
                                 unsigned int scale = 2);
        // Draw the object bounding boxes and labels on the image
        void drawBBoxLabel(cv::Mat &image, const Object &object,
                           const DetectionParams &params = DetectionParams(),
                           unsigned int scale = 2);
        void drawSegmentation(cv::Mat &mask, const Object &object);

        void readClassLabel(const std::string &label_file);

    protected:
        virtual std::vector<Object> postprocess(
            std::unordered_map<std::string, std::vector<float>> &feature_vector,
            const DetectionParams &params = DetectionParams(),
            const std::vector<std::string> &detected_class = {}) = 0;
        std::map<int, std::string> class_labels_;
        int CATEGORY;
        bool agnostic_;
        std::vector<cv::Scalar> class_colors_;

        const std::vector<std::vector<unsigned int>> KPS_COLORS = {
            {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};

        const std::vector<std::vector<unsigned int>> SKELETON = {
            {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};

        const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
            {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}};
    };
} // namespace tensorrt_inference