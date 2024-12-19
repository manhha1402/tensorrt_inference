#include <yaml-cpp/yaml.h>

#include "tensorrt_inference/tensorrt_inference.h"
// Runs object detection on video stream then displays annotated results.
int main(int argc, char *argv[])
{
  // Read the input image
  std::string inputImage = argv[1];
  auto img = cv::imread(inputImage);
  if (img.empty())
  {
    std::cout << "Error: Unable to read image at path '" << inputImage << "'"
              << std::endl;
    return -1;
  }
  std::string model_name = argv[2];
  std::shared_ptr<tensorrt_inference::Detection> detection;
  if (model_name.find("yolov8") != std::string::npos)
  {
    detection = std::make_shared<tensorrt_inference::YoloV8>(model_name);
  }
  else if (model_name.find("yolov9") != std::string::npos)
  {
    detection = std::make_shared<tensorrt_inference::YoloV9>(model_name);
  }
  else if (model_name.find("facedetector") != std::string::npos)
  {
    detection = std::make_shared<tensorrt_inference::RetinaFace>(model_name);
  }
  else if (model_name.find("retinaface") != std::string::npos)
  {
    detection = std::make_shared<tensorrt_inference::RetinaFace>(model_name);
  }
  // } else if (model_name.find("plate_detection") != std::string::npos) {
  //   detection = std::make_shared<tensorrt_inference::YoloV8>(model_name);
  // } else {
  //   std::cout << "unkown model" << std::endl;
  //   return 1;
  // }

  // detection = std::make_shared<tensorrt_inference::YoloV8>(model_name);

  // Run inference
  tensorrt_inference::DetectionParams params(0.5, 0.5, 0.5, 0.5, 20);
  std::vector<std::string> detected_classes{"all"};
  auto start = std::chrono::high_resolution_clock::now();
  const auto objects = detection->detect(img, params, detected_classes);
  auto end = std::chrono::high_resolution_clock::now();
  // Calculate elapsed time
  std::chrono::duration<double> elapsed = end - start;

  // Output elapsed time
  std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
  std::cout << "Detected " << objects.size() << " objects" << std::endl;

  // Draw the bounding boxes on the image
  auto result = detection->drawObjectLabels(img, objects);

  // Save the image to disk
  const auto outputName =
      inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
  cv::imwrite(outputName, result);
  std::cout << "Saved annotated image to: " << outputName << std::endl;

  return 0;
}