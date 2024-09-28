#pragma once
#include <fstream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "tensorrt_inference/types.h"
namespace tensorrt_inference {

inline std::map<int, std::string> readClassLabel(const std::string &fileName) {
  YAML::Node config = YAML::LoadFile(fileName);

  std::map<int, std::string> class_label;
  for (size_t i = 0; i < config.size(); ++i) {
    class_label[i] = config[i].as<std::string>();
  }
  return class_label;
}
inline std::vector<CroppedObject> getCroppedObjects(
    const cv::Mat &frame,
    const std::vector<tensorrt_inference::Object> &objects, const int w,
    const int h, bool resize = true) {
  std::vector<CroppedObject> cropped_objects;
  for (auto &object : objects) {
    cv::Mat tempCrop = frame(object.rect);
    CroppedObject curr_object;
    if (resize) {
      cv::resize(tempCrop, curr_object.croped_object, cv::Size(w, h), 0, 0,
                 cv::INTER_CUBIC);  // resize to network dimension input
    } else {
      curr_object.croped_object = tempCrop;
    }

    curr_object.rect = object.rect;
    cropped_objects.push_back(curr_object);
  }
  return cropped_objects;
}

// refenrence:
// https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

cv::cuda::GpuMat getRotateCropImage(const cv::cuda::GpuMat &srcImage,
                                    const std::vector<std::vector<int>> &box);

void resizeOp(const cv::cuda::GpuMat &img, cv::cuda::GpuMat &resize_img,
              float wh_ratio,
              const std::array<int, 3> &rec_image_shape = {3, 48, 320});

void normalizeOp(cv::cuda::GpuMat &im, const std::array<float, 3> &mean,
                 const std::array<float, 3> &scale);

void permute(cv::cuda::GpuMat &im);

void permuteBatchOp(const std::vector<cv::cuda::GpuMat> &imgs,
                    cv::cuda::GpuMat &dest);
}  // namespace tensorrt_inference