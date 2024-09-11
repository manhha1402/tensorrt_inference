#include "tensorrt_inference/tensorrt_inference.h"
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
    std::shared_ptr<tensorrt_inference::Detection> detection;
    if(model_name.find("yolov8") != std::string::npos)
    {
        detection = std::make_shared< tensorrt_inference::YoloV8>(model_path.string(),config);
    }
    else if (model_name.find("yolov9") != std::string::npos)
    {
        detection = std::make_shared< tensorrt_inference::YoloV9>(model_path.string(),config);
    }
    else if (model_name.find("face") != std::string::npos)
    {
        detection = std::make_shared< tensorrt_inference::RetinaFace>(model_path.string(),config);
    }
    else{
        std::cout<<"unkown model"<<std::endl;
        return 1;
    }
    // Run inference
    const auto objects = detection->detect(img);

    // Draw the bounding boxes on the image
    detection->drawObjectLabels(img, objects);

    std::cout << "Detected " << objects.size() << " objects" << std::endl;

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}