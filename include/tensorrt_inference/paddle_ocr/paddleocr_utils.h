#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#include "tensorrt_inference/paddle_ocr/clipper.h"
#include "tensorrt_inference/utils.h"
namespace tensorrt_inference
{

  void GetContourArea(const std::vector<std::vector<float>> &box,
                      float unclip_ratio, float &distance);

  cv::RotatedRect UnClip(std::vector<std::vector<float>> box,
                         const float &unclip_ratio);

  float **Mat2Vec(cv::Mat mat);

  std::vector<std::vector<int>> OrderPointsClockwise(
      std::vector<std::vector<int>> pts);

  std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, float &ssid);

  float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);
  float PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred);

  std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(
      const cv::Mat &pred, const cv::Mat &bitmap, const float &box_thresh,
      const float &det_db_unclip_ratio, const bool &use_polygon_score);

  std::vector<std::vector<std::vector<int>>> FilterTagDetRes(
      std::vector<std::vector<std::vector<int>>> &boxes, float ratio_h,
      float ratio_w, const cv::cuda::GpuMat &srcimg);

  bool XsortInt(std::vector<int> a, std::vector<int> b);

  bool XsortFp32(std::vector<float> a, std::vector<float> b);

  std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

  inline int _max(int a, int b) { return a >= b ? a : b; }

  inline int _min(int a, int b) { return a >= b ? b : a; }

  template <class T>
  inline T clamp(T x, T min, T max)
  {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }

  inline float clampf(float x, float min, float max)
  {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }

} // namespace tensorrt_inference