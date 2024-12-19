#include "tensorrt_inference/yolov8.h"
namespace tensorrt_inference
{
  YoloV8::YoloV8(const std::string &model_name,
                 tensorrt_inference::Options options,
                 const std::filesystem::path &model_dir)
      : Detection(model_name, options, model_dir) {}
  std::vector<Object> YoloV8::postprocess(
      std::unordered_map<std::string, std::vector<float>> &feature_vectors,
      const DetectionParams &params,
      const std::vector<std::string> &detected_class)
  {
    if (m_trtEngine->getOutputInfo().size() == 1)
    {
      if (m_trtEngine->getOutputInfo().at("output0").dims.d[1] == 56)
      {
        // human pose
        return postprocessPose(feature_vectors, params, detected_class);
      }
      else
      {
        return postprocessDetect(feature_vectors, params, detected_class);
      }
    }
    else if (m_trtEngine->getOutputInfo().size() == 2)
    {
      return postProcessSegmentation(feature_vectors, params, detected_class);
    }
    else
    {
      return std::vector<Object>();
    }
  }

  // Postprocess the output
  std::vector<Object> YoloV8::postprocessDetect(
      std::unordered_map<std::string, std::vector<float>> &feature_vectors,
      const DetectionParams &params,
      const std::vector<std::string> &detected_class)
  {
    int rows = m_trtEngine->getOutputInfo().at("output0").dims.d[2];
    int dimensions = m_trtEngine->getOutputInfo().at("output0").dims.d[1];
    cv::Mat outputs(dimensions, rows, CV_32F, feature_vectors.at("output0").data());
    outputs = outputs.reshape(1, dimensions);
    cv::transpose(outputs, outputs);

    std::vector<float> output_data(outputs.rows * outputs.cols);
    std::memcpy(output_data.data(), outputs.ptr<float>(), output_data.size() * sizeof(float));

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> labels;

    for (int i = 0; i < rows; ++i)
    {
      float *row_data = &output_data[i * dimensions]; // Access row data

      float *classes_scores = row_data + 4; // number of classes
      cv::Mat scores(1, class_labels_.size(), CV_32FC1, classes_scores);
      cv::Point class_id;
      double max_class_score;
      cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      if (max_class_score > params.obj_threshold)
      {
        if (detected_class.empty() ||
            (detected_class.size() == 1 &&
             (detected_class[0] == "all" || detected_class[0] == "")))
        {
          confidences.push_back(max_class_score);
          labels.push_back(class_id.x);
          float x = row_data[0];
          float y = row_data[1];
          float w = row_data[2];
          float h = row_data[3];

          int left = int((x - 0.5 * w) * ratios_[0]);
          int top = int((y - 0.5 * h) * ratios_[1]);
          int width = int(w * ratios_[0]);
          int height = int(h * ratios_[1]);
          bboxes.push_back(cv::Rect(left, top, width, height));
        }
        else
        {
          if (std::find(detected_class.begin(), detected_class.end(),
                        class_labels_[class_id.x]) != detected_class.end())
          {
            confidences.push_back(max_class_score);
            labels.push_back(class_id.x);
            float x = row_data[0];
            float y = row_data[1];
            float w = row_data[2];
            float h = row_data[3];

            int left = int((x - 0.5 * w) * ratios_[0]);
            int top = int((y - 0.5 * h) * ratios_[1]);
            int width = int(w * ratios_[0]);
            int height = int(h * ratios_[1]);
            bboxes.push_back(cv::Rect(left, top, width, height));
          }
        }
      }
      // data += dimensions;
    }
    std::vector<int> indices;
    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, confidences, labels, params.obj_threshold,
                             params.nms_threshold, indices);

    std::vector<Object> objects;
    // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices)
    {
      if (cnt >= params.num_detect)
      {
        break;
      }
      Object obj{};
      obj.probability = confidences[chosenIdx];
      obj.label = labels[chosenIdx];
      obj.rect = bboxes[chosenIdx];
      objects.push_back(obj);
      cnt += 1;
    }
    return objects;
  }

  // Postprocess the output for segmentation model
  std::vector<Object> YoloV8::postProcessSegmentation(
      std::unordered_map<std::string, std::vector<float>> &feature_vectors,
      const DetectionParams &params,
      const std::vector<std::string> &detected_class)
  {
    const auto &output0_info = m_trtEngine->getOutputInfo().at("output0");
    const auto &output1_info = m_trtEngine->getOutputInfo().at("output1");

    const int rows = output0_info.dims.d[2];
    const int dimensions = output0_info.dims.d[1];
    cv::Mat outputs(dimensions, rows, CV_32F, feature_vectors.at("output0").data());
    outputs = outputs.reshape(1, dimensions);
    cv::transpose(outputs, outputs);
    std::vector<float> output_data(outputs.rows * outputs.cols);
    std::memcpy(output_data.data(), outputs.ptr<float>(), output_data.size() * sizeof(float));

    const int num_net0_channels = output0_info.dims.d[1];
    const int num_seg_channels = output1_info.dims.d[1];

    const int num_net0_anchors = output0_info.dims.d[2];
    const int SEG_H = output1_info.dims.d[2];
    const int SEG_W = output1_info.dims.d[3];
    const auto numClasses = num_net0_channels - num_seg_channels - 4;
    cv::Mat protos = cv::Mat(num_seg_channels, SEG_H * SEG_W, CV_32F,
                             feature_vectors.at("output1").data());
    std::vector<int> labels;
    std::vector<float> confidences;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> maskConfs;
    std::vector<int> indices;

    // Object the bounding boxes and class labels
    for (int i = 0; i < rows; i++)
    {
      float *row_data = &output_data[i * dimensions]; // Access row data

      float *classes_scores = row_data + 4; // number of classes
      cv::Mat scores(1, class_labels_.size(), CV_32FC1, classes_scores);
      cv::Point class_id;
      double max_class_score;
      cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

      auto maskConfsPtr = row_data + 4 + numClasses;
      if (max_class_score > params.obj_threshold)
      {
        if (detected_class.empty() ||
            (detected_class.size() == 1 &&
             (detected_class[0] == "all" || detected_class[0] == "")))
        {
          confidences.push_back(max_class_score);
          labels.push_back(class_id.x);
          float x = row_data[0];
          float y = row_data[1];
          float w = row_data[2];
          float h = row_data[3];

          int left = int((x - 0.5 * w) * ratios_[0]);
          int top = int((y - 0.5 * h) * ratios_[1]);
          int width = int(w * ratios_[0]);
          int height = int(h * ratios_[1]);
          bboxes.push_back(cv::Rect(left, top, width, height));

          cv::Mat maskConf = cv::Mat(1, num_seg_channels, CV_32F, maskConfsPtr);
          maskConfs.push_back(maskConf);
        }
        else
        {
          if (std::find(detected_class.begin(), detected_class.end(),
                        class_labels_[class_id.x]) != detected_class.end())
          {
            confidences.push_back(max_class_score);
            labels.push_back(class_id.x);
            float x = row_data[0];
            float y = row_data[1];
            float w = row_data[2];
            float h = row_data[3];

            int left = int((x - 0.5 * w) * ratios_[0]);
            int top = int((y - 0.5 * h) * ratios_[1]);
            int width = int(w * ratios_[0]);
            int height = int(h * ratios_[1]);
            bboxes.push_back(cv::Rect(left, top, width, height));

            cv::Mat maskConf = cv::Mat(1, num_seg_channels, CV_32F, maskConfsPtr);
            maskConfs.push_back(maskConf);
          }
        }
      }
    }

    // Require OpenCV 4.7 for this function
    cv::dnn::NMSBoxesBatched(bboxes, confidences, labels, params.obj_threshold,
                             params.nms_threshold, indices);
    // Obtain the segmentation masks
    cv::Mat masks;
    std::vector<Object> objects;
    int cnt = 0;
    for (auto &i : indices)
    {
      if (cnt >= params.num_detect)
      {
        break;
      }
      cv::Rect tmp = bboxes[i];
      Object obj;
      obj.label = labels[i];
      obj.rect = tmp;
      obj.probability = confidences[i];
      masks.push_back(maskConfs[i]);
      objects.push_back(obj);
      cnt += 1;
    }

    // Convert segmentation mask to original frame
    if (!masks.empty())
    {
      cv::Mat matmulRes = (masks * protos).t();
      cv::Mat maskMat = matmulRes.reshape(indices.size(), {SEG_W, SEG_H});

      std::vector<cv::Mat> maskChannels;
      cv::split(maskMat, maskChannels);

      cv::Rect roi;
      if (input_frame_h_ > input_frame_w_)
      {
        roi = cv::Rect(0, 0, float(SEG_W * input_frame_w_) / float(input_frame_h_), SEG_H);
      }
      else
      {
        roi = cv::Rect(0, 0, SEG_W, float(SEG_H * input_frame_h_) / float(input_frame_w_));
      }
      for (size_t i = 0; i < indices.size(); i++)
      {
        cv::Mat dest, mask;
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);
        dest = dest(roi);
        cv::resize(dest, mask,
                   cv::Size(static_cast<int>(input_frame_w_),
                            static_cast<int>(input_frame_h_)),
                   cv::INTER_LINEAR);
        objects[i].box_mask = mask(objects[i].rect) > params.seg_threshold;
      }
    }
    return objects;
  }

  std::vector<Object> YoloV8::postprocessPose(
      std::unordered_map<std::string, std::vector<float>> &feature_vectors,
      const DetectionParams &params,
      const std::vector<std::string> &detected_class)
  {
    int rows = m_trtEngine->getOutputInfo().at("output0").dims.d[2];
    int dimensions = m_trtEngine->getOutputInfo().at("output0").dims.d[1];
    cv::Mat outputs(dimensions, rows, CV_32F, feature_vectors.at("output0").data());
    outputs = outputs.reshape(1, dimensions);
    cv::transpose(outputs, outputs);
    // float *data = (float *)outputs.data;
    std::vector<float> output_data(outputs.rows * outputs.cols);
    std::memcpy(output_data.data(), outputs.ptr<float>(), output_data.size() * sizeof(float));

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> labels;
    std::vector<std::vector<float>> kpss;

    for (int i = 0; i < rows; ++i)
    {
      float *row_data = &output_data[i * dimensions]; // Access row data

      float *classes_scores = row_data + 4; // number of classes
      auto kps_ptr = row_data + 5;
      cv::Point class_id;
      float max_class_score = *classes_scores;
      if (max_class_score > params.obj_threshold)
      {
        if (detected_class.empty() ||
            (detected_class.size() == 1 &&
             (detected_class[0] == "all" || detected_class[0] == "")))
        {
          confidences.push_back(max_class_score);
          labels.push_back(class_id.x);
          float x = row_data[0];
          float y = row_data[1];
          float w = row_data[2];
          float h = row_data[3];

          int left = int((x - 0.5 * w) * ratios_[0]);
          int top = int((y - 0.5 * h) * ratios_[1]);
          int width = int(w * ratios_[0]);
          int height = int(h * ratios_[1]);
          bboxes.push_back(cv::Rect(left, top, width, height));

          std::vector<float> kps;
          for (int k = 0; k < num_kps_; k++)
          {
            float kpsX = *(kps_ptr + 3 * k) * ratios_[0];
            float kpsY = *(kps_ptr + 3 * k + 1) * ratios_[1];
            float kpsS = *(kps_ptr + 3 * k + 2);
            kpsX = std::clamp(kpsX, 0.f, float(input_frame_w_));
            kpsY = std::clamp(kpsY, 0.f, float(input_frame_h_));
            kps.push_back(kpsX);
            kps.push_back(kpsY);
            kps.push_back(kpsS);
          }
          kpss.push_back(kps);
        }
        else
        {
          class_id.x = 0; // person
          if (std::find(detected_class.begin(), detected_class.end(),
                        class_labels_[class_id.x]) != detected_class.end())
          {
            confidences.push_back(max_class_score);
            labels.push_back(class_id.x);
            float x = row_data[0];
            float y = row_data[1];
            float w = row_data[2];
            float h = row_data[3];

            int left = int((x - 0.5 * w) * ratios_[0]);
            int top = int((y - 0.5 * h) * ratios_[1]);
            int width = int(w * ratios_[0]);
            int height = int(h * ratios_[1]);
            bboxes.push_back(cv::Rect(left, top, width, height));
            std::vector<float> kps;
            for (int k = 0; k < num_kps_; k++)
            {
              float kpsX = *(kps_ptr + 3 * k) * ratios_[0];
              float kpsY = *(kps_ptr + 3 * k + 1) * ratios_[1];
              float kpsS = *(kps_ptr + 3 * k + 2);
              kpsX = std::clamp(kpsX, 0.f, float(input_frame_w_));
              kpsY = std::clamp(kpsY, 0.f, float(input_frame_h_));
              kps.push_back(kpsX);
              kps.push_back(kpsY);
              kps.push_back(kpsS);
            }
            kpss.push_back(kps);
          }
        }
      }
    }
    std::vector<int> indices;

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, confidences, labels, params.obj_threshold,
                             params.nms_threshold, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices)
    {
      if (cnt >= params.num_detect)
      {
        break;
      }

      Object obj{};
      obj.probability = confidences[chosenIdx];
      obj.label = labels[chosenIdx];
      obj.rect = bboxes[chosenIdx];
      obj.kps = kpss[chosenIdx];
      objects.push_back(obj);

      cnt += 1;
    }
    return objects;
  }

} // namespace tensorrt_inference