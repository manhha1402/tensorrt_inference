#include "tensorrt_inference/retinaface.h"
#include <numeric>


// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    
    const std::string home_dir = std::getenv("HOME");
    std::filesystem::path model_path = std::filesystem::path(home_dir) / "data" / "weights" / "retinaface";
    std::string config_file = model_path.string() + "/config.yaml";
    YAML::Node config = YAML::LoadFile(config_file);
    tensorrt_inference::RetinaFace retinaface(model_path.string(),config);
     // Read the input image
    std::string inputImage = argv[1];
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    const auto faces = retinaface.detectFaces(img);

    // // Draw the bounding boxes on the image
    // yolo9.drawObjectLabels(img, objects);

    // std::cout << "Detected " << objects.size() << " objects" << std::endl;

    // // Save the image to disk
    // const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    // cv::imwrite(outputName, img);
    // std::cout << "Saved annotated image to: " << outputName << std::endl;

    // return 0;
}