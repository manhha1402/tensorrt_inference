#include "tensorrt_inference/yolov8.h"


// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    tensorrt_inference::YoloV8Config config;
    std::string inputImage = argv[2];
    const std::string home_dir = std::getenv("HOME");
    std::filesystem::path model_path = std::filesystem::path(std::string(std::getenv("HOME"))) / "data" / "weights";

    // Create the YoloV8 engine
    tensorrt_inference::YoloV8 yoloV8(model_path.string(),argv[1], config);
    std::string class_name  =  "person";
    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    const auto objects = yoloV8.detectObjects(img);

    // Draw the bounding boxes on the image
    yoloV8.drawObjectLabels(img, objects,{class_name});

    std::cout << "Detected " << objects.size() << " objects" << std::endl;

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}