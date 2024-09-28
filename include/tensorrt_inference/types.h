#pragma once
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <opencv2/opencv.hpp>
namespace tensorrt_inference {

struct DetectionParams {
  float obj_threshold;
  float nms_threshold;
  float seg_threshold;
  float kps_threshold;
  int num_detect;

  DetectionParams()
      : obj_threshold(0.25f),
        nms_threshold(0.65f),
        seg_threshold(0.5),
        kps_threshold(0.5),
        num_detect(20) {}
  DetectionParams(const float obj_threshold, const float nms_threshold,
                  const float seg_threshold, const float kps_threshold,
                  const int num_detect)
      : obj_threshold(obj_threshold),
        nms_threshold(nms_threshold),
        seg_threshold(seg_threshold),
        kps_threshold(kps_threshold),
        num_detect(num_detect) {}
};

struct Object {
  // The object class.
  int label{};
  // The detection's confidence probability.
  float probability{};
  // The object bounding box rectangle.
  cv::Rect rect;
  // Semantic segmentation mask
  cv::Mat box_mask;
  // Pose estimation key points
  std::vector<float> kps{};
  // Landmarks face detection
  std::vector<cv::Point2f> landmarks{};
};
struct CroppedObject {
  double probability;
  std::string label;
  cv::Rect rect;
  cv::Mat croped_object;
};

}  // namespace tensorrt_inference