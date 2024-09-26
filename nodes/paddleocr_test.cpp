#include <yaml-cpp/yaml.h>

#include <opencv2/cudaimgproc.hpp>

#include "tensorrt_inference/tensorrt_inference.h"

// Runs object detection on video stream then displays annotated results.
int main(int argc, char *argv[]) {
  // Read the input image
  std::string inputImage = argv[1];
  auto img = cv::imread(inputImage);
  if (img.empty()) {
    std::cout << "Error: Unable to read image at path '" << inputImage << "'"
              << std::endl;
    return -1;
  }
  std::shared_ptr<tensorrt_inference::Detection> detection;
  detection = std::make_shared<tensorrt_inference::YoloV8>("plate_detection");

  // Run inference
  tensorrt_inference::DetectionParams params(0.5, 0.5, 0.5, 0.5, 20);
  std::vector<std::string> detected_classes{"all"};
  const auto objects = detection->detect(img, params, detected_classes);
  std::cout << "Detected " << objects.size() << " objects" << std::endl;

  // Draw the bounding boxes on the image
  auto result = detection->drawObjectLabels(img, objects);

  // Save the image to disk
  const auto outputName =
      inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
  cv::imwrite(outputName, result);
  std::cout << "Saved annotated image to: " << outputName << std::endl;
  auto plates = tensorrt_inference::getCroppedObjects(img, objects, img.cols,
                                                      img.rows, false);
  int i = 0;
  for (const auto &plate : plates) {
    cv::imwrite("cropped_" + std::to_string(i) + ".jpg", plate.croped_object);
    i++;
  }
  const std::filesystem::path &model_dir =
      std::filesystem::path(std::getenv("HOME")) / "data" / "weights";
  const std::string model_name = "paddleocr";
  tensorrt_inference::Options options;
  options.MIN_DIMS_ = {1, 3, 48, 10};
  options.OPT_DIMS_ = {1, 3, 48, 320};
  options.MAX_DIMS_ = {8, 3, 48, 2000};
  options.engine_file_dir = (model_dir / model_name).string();
  auto rec =
      std::make_shared<tensorrt_inference::PaddleOCR>(model_name, options);
  for (size_t i = 0; i < plates.size(); i++) {
    cv::cuda::GpuMat image(plates[i].croped_object);

    cv::cuda::GpuMat input = rec->preprocess(image);
    std::unordered_map<std::string, std::vector<float>> feature_vectors;
    rec->runInference(input, feature_vectors);
  }
  return 0;
}