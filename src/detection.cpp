

#include "tensorrt_inference/detection.h"
#include "tensorrt_inference/label_const.h"
namespace tensorrt_inference
{
  Detection::Detection(const std::string &model_name,
                       tensorrt_inference::Options options,
                       const std::filesystem::path &model_dir)
      : Model(model_name, options, model_dir)
  {

    class_labels_ = class_labels;
    CATEGORY = class_labels_.size();
    class_colors_.resize(CATEGORY);
    srand((int)time(nullptr));
    for (cv::Scalar &class_color : class_colors_)
      class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
  }

  void Detection::readClassLabel(const std::string &label_file)
  {
    class_labels_.clear();
    YAML::Node config = YAML::LoadFile(label_file);
    for (size_t i = 0; i < config.size(); ++i)
    {
      class_labels_[i] = config[i].as<std::string>();
    }
    CATEGORY = class_labels_.size();
    class_colors_.resize(CATEGORY);
    srand((int)time(nullptr));
    for (cv::Scalar &class_color : class_colors_)
      class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
  }

  std::vector<Object> Detection::detect(
      const cv::Mat &inputImageBGR, const DetectionParams &params,
      const std::vector<std::string> &detected_class)
  {
    std::unordered_map<std::string, std::vector<float>> feature_vectors;
    bool res = doInference(inputImageBGR, feature_vectors);
    std::vector<Object> ret =
        postprocess(feature_vectors, params, detected_class);
    return ret;
  }

  void Detection::drawBBoxLabel(cv::Mat &image, const Object &object,
                                const DetectionParams &params,
                                unsigned int scale)
  {
    // Choose the color
    int colorIndex =
        object.label %
        class_colors_.size(); // We have only defined 80 unique colors
    cv::Scalar color =
        cv::Scalar(class_colors_[colorIndex][0], class_colors_[colorIndex][1],
                   class_colors_[colorIndex][2]);
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
    sprintf(text, "%s %.1f%%", class_labels_[object.label].c_str(),
            object.probability * 100);

    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                         0.35 * scale, scale, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7 * 255;

    int x = object.rect.x;
    int y = object.rect.y + 1;

    cv::rectangle(image, rect, color * 255, scale + 1);

    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(labelSize.width, labelSize.height + baseLine)),
        txt_bk_color, -1);

    cv::putText(image, text, cv::Point(x, y + labelSize.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

    // Pose estimation
    if (!object.kps.empty())
    {
      auto &kps = object.kps;
      for (int k = 0; k < num_kps_ + 2; k++)
      {
        if (k < num_kps_)
        {
          int kpsX = std::round(kps[k * 3]);
          int kpsY = std::round(kps[k * 3 + 1]);
          float kpsS = kps[k * 3 + 2];
          if (kpsS > params.kps_threshold)
          {
            cv::Scalar kpsColor =
                cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
            cv::circle(image, {kpsX, kpsY}, 5, kpsColor, -1);
          }
        }
        auto &ske = SKELETON[k];
        int pos1X = std::round(kps[(ske[0] - 1) * 3]);
        int pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);

        int pos2X = std::round(kps[(ske[1] - 1) * 3]);
        int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);

        float pos1S = kps[(ske[0] - 1) * 3 + 2];
        float pos2S = kps[(ske[1] - 1) * 3 + 2];

        if (pos1S > params.kps_threshold && pos2S > params.kps_threshold)
        {
          cv::Scalar limbColor =
              cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
          cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
        }
      }
    }
    if (!object.landmarks.empty())
    {
      for (int kp = 0; kp < object.landmarks.size(); kp++)
      {
        cv::Scalar kpsColor = cv::Scalar(255, 0, 0);
        cv::circle(image, object.landmarks[kp], 5, kpsColor, -1);
      }
    }
  }
  void Detection::drawSegmentation(cv::Mat &mask, const Object &object)
  {
    // Choose the color
    int colorIndex =
        object.label %
        class_colors_.size(); // We have only defined 80 unique colors
    // Add the mask for said object
    mask(object.rect).setTo(class_colors_[colorIndex] * 255, object.box_mask);
  }

  cv::Mat Detection::drawObjectLabels(const cv::Mat &image,
                                      const std::vector<Object> &objects,
                                      const DetectionParams &params,
                                      unsigned int scale)
  {
    cv::Mat result = image.clone();
    //  If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].box_mask.empty())
    {
      cv::Mat mask = result.clone();
      for (const auto &object : objects)
      {
        drawSegmentation(result, object);
      }
      // Add all the masks to our image
      cv::addWeighted(result, 0.5, mask, 0.8, 1, result);
    }
    // Bounding boxes and annotations
    for (auto &object : objects)
    {
      drawBBoxLabel(result, object, params, scale);
    }
    return result;
  }
} // namespace tensorrt_inference