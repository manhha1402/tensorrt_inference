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

    // Run inference
    const auto faces = retinaface.detectFaces(img);
    for(const auto &face : faces) {
        cv::Scalar color= cv::Scalar(255, 0, 0);
        cv::rectangle(img, face.rect, color, 2, cv::LINE_8, 0);

    }
    cv::imwrite("result.jpg",img);

    return 0;
}