#include "cmd_line_util.h"
#include "tensorrt_inference/yolov9.h"

// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    tensorrt_inference::YoloV9Config config;
    std::string onnxModelPath;
    std::string inputImage;

    // Parse the command line arguments
    if (!tensorrt_inference::parseArguments(argc, argv, config, onnxModelPath, inputImage)) {
        return -1;
    }

    // Create the YoloV9 engine
    tensorrt_inference::YoloV9 yoloV9(onnxModelPath, config);

    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    const auto objects = yoloV9.detectObjects(img);

    // Draw the bounding boxes on the image
    yoloV9.drawObjectLabels(img, objects);

    std::cout << "Detected " << objects.size() << " objects" << std::endl;

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}