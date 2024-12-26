#include <yaml-cpp/yaml.h>

#include "tensorrt_inference/tensorrt_inference.h"
// Runs object detection on video stream then displays annotated results.
int main(int argc, char *argv[])
{
  // Read the input image
  std::string inputImage = argv[1];
  std::string model_name = argv[2];
  auto img = cv::imread(inputImage);
  if (img.empty())
  {
    std::cout << "Error: Unable to read image at path '" << inputImage << "'"
              << std::endl;
    return -1;
  }
  tensorrt_inference::Model model(model_name);
}