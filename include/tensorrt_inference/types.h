#pragma once
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <opencv2/opencv.hpp>
namespace tensorrt_inference
{

  struct PreprocessParams
  {
    cv::Scalar sub_vals;
    cv::Scalar div_vals;
    bool normalized;
    bool swapBR;
    bool keep_ratio;
    PreprocessParams() : sub_vals(cv::Scalar(0, 0, 0)),
                         div_vals(cv::Scalar(1.0, 1.0, 1.0)),
                         normalized(true),
                         swapBR(true),
                         keep_ratio(true)
    {
    }
    PreprocessParams(const cv::Scalar &sub_vals,
                     const cv::Scalar &div_vals,
                     const bool normalized,
                     const bool swapBR,
                     const bool keep_ratio)
        : sub_vals(sub_vals), div_vals(div_vals), normalized(normalized), swapBR(swapBR), keep_ratio(keep_ratio)
    {
    }
    // Overload = as a member function
    PreprocessParams operator=(const PreprocessParams &other)
    {
      if (this == &other)
      {
        // Check for self-assignment
        return *this;
      }
      sub_vals = other.sub_vals;
      div_vals = other.div_vals;
      normalized = other.normalized;
      swapBR = other.swapBR;
      keep_ratio = other.keep_ratio;
      return *this;
    }
    // print out
    // Overload the stream insertion operator for the struct
    void printInfo()
    {
      std::cout << "sub_vals: " << sub_vals << std::endl;
      std::cout << "div_vals: " << div_vals << std::endl;
      std::cout << "normalized: " << normalized << std::endl;
      std::cout << "swapBR: " << swapBR << std::endl;
      std::cout << "keep_ratio: " << keep_ratio << std::endl;
    }
  };

  struct DetectionParams
  {
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

  struct Object
  {
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
  struct CroppedObject
  {
    double det_score;
    double rec_score;
    std::string label;
    cv::Rect rect;
    cv::Mat croped_object;
    cv::Mat original_image;
  };

} // namespace tensorrt_inference