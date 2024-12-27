#include <yaml-cpp/yaml.h>

#include <opencv2/cudaimgproc.hpp>

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
  std::shared_ptr<tensorrt_inference::Detection> detection;
  detection =
      std::make_shared<tensorrt_inference::YoloV8>("plate_yolov8n_320_2024");
  std::string home_dir = std::getenv("HOME");
  std::string lable_file = home_dir + "/data/weights/plate_yolov8n_320_2024/plate_yolov8n_320_2024.yaml";
  detection->readClassLabel(lable_file);
  // Run inference
  tensorrt_inference::DetectionParams params(0.3, 0.5, 0.5, 0.5, 1);
  std::vector<std::string> detected_classes{"all"};
  const auto objects = detection->detect(img, params, detected_classes);
  std::cout << "Detected " << objects.size() << " objects" << std::endl;

  // Draw the bounding boxes on the image
  auto result = detection->drawObjectLabels(img, objects);

  // Save the image to disk
  auto outputName =
      inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
  cv::imwrite(outputName, result);
  std::cout << "Saved annotated image to: " << outputName << std::endl;
  auto plates = tensorrt_inference::getCroppedObjects(img, objects, img.cols,
                                                      img.rows, false);
  int k = 0;
  for (const auto &plate : plates)
  {
    cv::imwrite("cropped_" + std::to_string(k) + ".jpg", plate.croped_object);
    k++;
  }

  const std::filesystem::path &model_dir =
      std::filesystem::path(std::getenv("HOME")) / "data" / "weights";
  const std::string model_name = "paddleocr";
  tensorrt_inference::Options options_rec;
  options_rec.MIN_DIMS_ = {1, 3, 48, 10};
  options_rec.OPT_DIMS_ = {1, 3, 48, 320};
  options_rec.MAX_DIMS_ = {8, 3, 48, 2000};
  options_rec.engine_file_dir = (model_dir / model_name).string();
  tensorrt_inference::Options options_det;
  options_det.engine_file_dir = (model_dir / model_name).string();
  auto paddle_ocr = std::make_shared<tensorrt_inference::PaddleOCR>(
      model_name, options_det, options_rec, model_dir);
  for (size_t i = 0; i < plates.size(); i++)
  {
    auto result = paddle_ocr->runInference(img, plates[i]);
    plates[i].rec_score = result.second;
    plates[i].label = result.first;
    std::cout << result.first << std::endl;
    std::cout << "detection score: " << result.second << std::endl;
    std::cout << "recognition score: " << plates[i].det_score << std::endl;
  }
  plates.erase(
      std::remove_if(plates.begin(), plates.end(),
                     [&](const tensorrt_inference::CroppedObject &plate)
                     {
                       return plate.rec_score < 0.8;
                     }),
      plates.end());
  if (!plates.empty())
  {
    cv::Mat rec = tensorrt_inference::drawBBoxLabels(img, plates, 1);
    cv::imshow("result", rec);
    cv::waitKey(0);
    outputName =
        inputImage.substr(0, inputImage.find_last_of('.')) + "_rec.jpg";
    cv::imwrite(outputName, rec);
  }
  else
  {
    std::cout << "no detection" << std::endl;
  }

  return 0;
}