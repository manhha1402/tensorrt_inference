#include "tensorrt_inference/yolov8.h"
#include <opencv2/cudaimgproc.hpp>
#include <yaml-cpp/yaml.h>
// Runs object detection on video stream then displays annotated results.
int main(int argc, char *argv[]) {
    const std::string home_dir = std::getenv("HOME");
     // Read the input image
    std::string inputImage = argv[1];
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }
    std::string model_name = argv[2];
    std::filesystem::path model_path = std::filesystem::path(std::string(std::getenv("HOME"))) / "data" / "weights" / model_name;
    std::string config_file = model_path.string() + "/config.yaml";
    YAML::Node config = YAML::LoadFile(config_file);
    tensorrt_inference::YoloV8 yolo8(model_path.string(),config);
   

    // Run inference
    const auto objects = yolo8.detectObjects(img);

    // Draw the bounding boxes on the image
    yolo8.drawObjectLabels(img, objects);

    std::cout << "Detected " << objects.size() << " objects" << std::endl;

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}