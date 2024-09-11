#include "tensorrt_inference/retinaface.h"
#include <numeric>


// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    
    const std::string home_dir = std::getenv("HOME");
    std::string inputImage = argv[1];
    std::string model_name = argv[2];
    std::filesystem::path model_path = std::filesystem::path(home_dir) / "data" / "weights" / model_name;
    std::string config_file = model_path.string() + "/config.yaml";
    YAML::Node config = YAML::LoadFile(config_file);
    tensorrt_inference::RetinaFace retinaface(model_path.string(),config);
     // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }
    //tensorrt_inference::ModelParams params(0.2,0.2,20);
    // Run inference
    const auto faces = retinaface.detect(img);
    retinaface.drawObjectLabels(img,faces);
   
    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}