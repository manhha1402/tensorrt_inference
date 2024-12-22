#include "tensorrt_inference/utils.h"

namespace tensorrt_inference
{

  cv::Mat drawBBoxLabels(const cv::Mat &image,
                         const std::vector<CroppedObject> &objects,
                         unsigned int scale, bool show_rec)
  {
    cv::Mat result = image.clone();
    for (const auto &object : objects)
    {
      cv::Scalar color = cv::Scalar(0, 255, 0);
      float meanColor = cv::mean(color)[0];
      cv::Scalar txtColor;
      if (meanColor > 0.5)
      {
        txtColor = cv::Scalar(0, 0, 0);
      }
      else
      {
        txtColor = cv::Scalar(255, 255, 255);
      }
      const auto &rect = object.rect;
      // Draw rectangles and text
      char text[256];
      if (show_rec)
      {
        sprintf(text, "%s %.1f%%", object.label.c_str(), object.rec_score * 100);
      }
      else
      {
        sprintf(text, "%s %.1f%%", object.label.c_str(), object.det_score * 100);
      }

      int baseLine = 0;
      cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                           0.35 * scale, scale, &baseLine);

      cv::Scalar txt_bk_color = color * 0.7 * 255;

      int x = object.rect.x;
      int y = object.rect.y + 1;

      cv::rectangle(result, rect, color * 255, scale + 1);

      cv::rectangle(
          result,
          cv::Rect(cv::Point(x, y),
                   cv::Size(labelSize.width, labelSize.height + baseLine)),
          txt_bk_color, -1);

      cv::putText(result, text, cv::Point(x, y + labelSize.height),
                  cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);
    }
    return result;
  }

  // refenrence:
  // https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
  cv::cuda::GpuMat getRotateCropImage(const cv::cuda::GpuMat &srcImage,
                                      const std::vector<std::vector<int>> &box)
  {
    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    int img_crop_width =
        int(sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2)));
    int img_crop_height =
        int(sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(box[0][0], box[0][1]);
    pointsf[1] = cv::Point2f(box[1][0], box[1][1]);
    pointsf[2] = cv::Point2f(box[2][0], box[2][1]);
    pointsf[3] = cv::Point2f(box[3][0], box[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::cuda::GpuMat dst_img;
    cv::cuda::warpPerspective(srcImage, dst_img, M,
                              cv::Size(img_crop_width, img_crop_height),
                              cv::BORDER_REPLICATE);

    cv::cuda::GpuMat res;
    if (float(dst_img.rows) >= float(dst_img.cols) * 3)
    { // 1.5
      cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
      cv::transpose(dst_img, srcCopy);
      cv::flip(srcCopy, srcCopy, 0);
      res.upload(srcCopy);
      return res;
    }
    else
    {
      return dst_img;
    }
  }

  void resizeOp(const cv::cuda::GpuMat &img, cv::cuda::GpuMat &resize_img,
                float wh_ratio, const std::array<int, 3> &rec_image_shape)
  {
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    imgW = int(48 * wh_ratio);

    float ratio = float(img.cols) / float(img.rows);
    int resize_w, resize_h;

    if (ceilf(imgH * ratio) > imgW)
      resize_w = imgW;
    else
      resize_w = int(ceilf(imgH * ratio));
    cv::cuda::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
                     cv::INTER_LINEAR);
    cv::cuda::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                             int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                             {127, 127, 127});
  }

  void normalizeOp(cv::cuda::GpuMat &im, const std::array<float, 3> &mean,
                   const std::array<float, 3> &scale)
  {
    im.convertTo(im, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(im, cv::Scalar(mean[0], mean[1], mean[2]), im,
                       cv::noArray(), -1);
    cv::cuda::multiply(im, cv::Scalar(scale[0], scale[1], scale[2]), im, 1, -1);
  }

  void permuteBatchOp(const std::vector<cv::cuda::GpuMat> &imgs,
                      cv::cuda::GpuMat &dest)
  {
    dest =
        cv::cuda::GpuMat(imgs.size(), imgs[0].rows * imgs[0].cols * 3, CV_32FC1);
    for (int j = 0; j < imgs.size(); j++)
    {
      int rh = imgs[j].rows;
      int rw = imgs[j].cols;
      size_t width = rh * rw;
      size_t start = width * 4 * 3 * j;
      std::vector<cv::cuda::GpuMat> input_channels{
          cv::cuda::GpuMat(rh, rw, CV_32FC1, &(dest.ptr()[start])),
          cv::cuda::GpuMat(rh, rw, CV_32FC1, &(dest.ptr()[start + width * 4])),
          cv::cuda::GpuMat(rh, rw, CV_32FC1, &(dest.ptr()[start + width * 8]))};

      cv::cuda::split(imgs[j], input_channels);
    }
  }
  void permute(cv::cuda::GpuMat &im)
  {
    cv::cuda::GpuMat tmp = im.clone();
    size_t width = im.rows * im.cols;
    std::vector<cv::cuda::GpuMat> input_channels{
        cv::cuda::GpuMat(im.rows, im.cols, CV_32FC1, &(im.ptr()[0])),
        cv::cuda::GpuMat(im.rows, im.cols, CV_32FC1, &(im.ptr()[width * 4])),
        cv::cuda::GpuMat(im.rows, im.cols, CV_32FC1, &(im.ptr()[width * 8]))};
    cv::cuda::split(tmp, input_channels); // HWC -> CHW
  }

} // namespace tensorrt_inference