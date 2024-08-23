#include "tensorrt_inference/yolo9_refactor.h"
#include <opencv2/cudaimgproc.hpp>
#include <yaml-cpp/yaml.h>
// Runs object detection on video stream then displays annotated results.
int main(int argc, char *argv[]) {
    const std::string home_dir = std::getenv("HOME");
    std::filesystem::path model_path = std::filesystem::path(std::string(std::getenv("HOME"))) / "data" / "weights" / "yolov9e";
    std::string config_file = model_path.string() + "/config.yaml";
    YAML::Node config = YAML::LoadFile(config_file);
    tensorrt_inference::YoloV9Refactor yolo9(model_path.string(),config);
     // Read the input image
    std::string inputImage = argv[2];
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    const auto objects = yolo9.detectObjects(img);

    // Draw the bounding boxes on the image
    yolo9.drawObjectLabels(img, objects);

    std::cout << "Detected " << objects.size() << " objects" << std::endl;

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}