#include "tensorrt_inference/yolov8.h"
namespace tensorrt_inference {
YoloV8::YoloV8(const std::string &model_name,
               tensorrt_inference::Options options,
               const std::filesystem::path &model_dir)
    : Detection(model_name, options, model_dir) {}
std::vector<Object> YoloV8::postprocess(
    std::unordered_map<std::string, std::vector<float>> &feature_vectors,
    const DetectionParams &params,
    const std::vector<std::string> &detected_class) {
  if (m_trtEngine->getOutputInfo().size() == 1) {
    if (m_trtEngine->getOutputInfo().at("output0").dims.d[1] == 56) {
      // human pose
      return postprocessPose(feature_vectors, params, detected_class);
    } else {
      return postprocessDetect(feature_vectors, params, detected_class);
    }
  } else if (m_trtEngine->getOutputInfo().size() == 2) {
    return postProcessSegmentation(feature_vectors, params, detected_class);
  } else {
    return std::vector<Object>();
  }
}

// Postprocess the output
std::vector<Object> YoloV8::postprocessDetect(
    std::unordered_map<std::string, std::vector<float>> &feature_vectors,
    const DetectionParams &params,
    const std::vector<std::string> &detected_class) {
  int num_anchors = m_trtEngine->getOutputInfo().at("output0").dims.d[2];
  int num_channels = m_trtEngine->getOutputInfo().at("output0").dims.d[1];
  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<int> indices;
  // Get all the YOLO proposals
  cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F,
                           feature_vectors.at("output0").data());
  output = output.t();
  for (int i = 0; i < num_anchors; i++) {
    auto rowPtr = output.row(i).ptr<float>();
    auto bboxesPtr = rowPtr;
    auto scoresPtr = rowPtr + 4;
    auto maxSPtr = std::max_element(scoresPtr, scoresPtr + CATEGORY);
    float score = *maxSPtr;
    if (score > params.obj_threshold) {
      if (detected_class.empty() ||
          (detected_class.size() == 1 &&
           (detected_class[0] == "all" || detected_class[0] == ""))) {
        int label = maxSPtr - scoresPtr;
        float x = *bboxesPtr++;
        float y = *bboxesPtr++;
        float w = *bboxesPtr++;
        float h = *bboxesPtr;

        float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, input_frame_w_);
        float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, input_frame_h_);
        float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, input_frame_w_);
        float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, input_frame_h_);

        cv::Rect_<float> bbox;
        bbox.x = x0;
        bbox.y = y0;
        bbox.width = x1 - x0;
        bbox.height = y1 - y0;
        bboxes.push_back(bbox);
        labels.push_back(label);
        scores.push_back(score);
      } else {
        int label = maxSPtr - scoresPtr;
        if (std::find(detected_class.begin(), detected_class.end(),
                      class_labels_[label]) != detected_class.end()) {
          float x = *bboxesPtr++;
          float y = *bboxesPtr++;
          float w = *bboxesPtr++;
          float h = *bboxesPtr;

          float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, input_frame_w_);
          float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, input_frame_h_);
          float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, input_frame_w_);
          float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, input_frame_h_);

          cv::Rect_<float> bbox;
          bbox.x = x0;
          bbox.y = y0;
          bbox.width = x1 - x0;
          bbox.height = y1 - y0;
          bboxes.push_back(bbox);
          labels.push_back(label);
          scores.push_back(score);
        }
      }
    }
  }
  // Run NMS
  cv::dnn::NMSBoxesBatched(bboxes, scores, labels, params.obj_threshold,
                           params.nms_threshold, indices);

  std::vector<Object> objects;
  // Choose the top k detections
  int cnt = 0;
  for (auto &chosenIdx : indices) {
    if (cnt >= params.num_detect) {
      break;
    }
    Object obj{};
    obj.probability = scores[chosenIdx];
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
    const std::vector<std::string> &detected_class) {
  std::cout<<"postProcessSegmentation"<<std::endl;
  const auto &output0_info = m_trtEngine->getOutputInfo().at("output0");
  const auto &output1_info = m_trtEngine->getOutputInfo().at("output1");

  const int num_net0_channels = output0_info.dims.d[1];
  const int num_seg_channels = output1_info.dims.d[1];

  const int num_net0_anchors = output0_info.dims.d[2];
  const int SEG_H = output1_info.dims.d[2];
  const int SEG_W = output1_info.dims.d[3];
  std::cout<<SEG_H<<std::endl;
  std::cout<<SEG_W<<std::endl;
  std::cout<<num_net0_channels<<std::endl;
  std::cout<<num_seg_channels<<std::endl;
  std::cout<<num_net0_anchors<<std::endl;

  const auto numClasses = num_net0_channels - num_seg_channels - 4;
  cv::Mat output = cv::Mat(num_net0_channels, num_net0_anchors, CV_32F,
                           feature_vectors.at("output0").data());
  output = output.t();

  cv::Mat protos = cv::Mat(num_seg_channels, SEG_H * SEG_W, CV_32F,
                           feature_vectors.at("output1").data());
  cv::imwrite("protos.jpg",protos);
  std::vector<int> labels;
  std::vector<float> scores;
  std::vector<cv::Rect> bboxes;
  std::vector<cv::Mat> maskConfs;
  std::vector<int> indices;

  // Object the bounding boxes and class labels
  for (int i = 0; i < num_net0_anchors; i++) {
    auto rowPtr = output.row(i).ptr<float>();
    auto bboxesPtr = rowPtr;
    auto scoresPtr = rowPtr + 4;
    auto maskConfsPtr = rowPtr + 4 + numClasses;
    auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
    float score = *maxSPtr;
    if (score > params.obj_threshold) {
      if (detected_class.empty() ||
          (detected_class.size() == 1 &&
           (detected_class[0] == "all" || detected_class[0] == ""))) {
        float x = *bboxesPtr++;
        float y = *bboxesPtr++;
        float w = *bboxesPtr++;
        float h = *bboxesPtr;

        float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, input_frame_w_);
        float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, input_frame_h_);
        float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, input_frame_w_);
        float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, input_frame_h_);

        int label = maxSPtr - scoresPtr;
        cv::Rect_<float> bbox;
        bbox.x = x0;
        bbox.y = y0;
        bbox.width = x1 - x0;
        bbox.height = y1 - y0;

        cv::Mat maskConf = cv::Mat(1, num_seg_channels, CV_32F, maskConfsPtr);

        bboxes.push_back(bbox);
        labels.push_back(label);
        scores.push_back(score);
        maskConfs.push_back(maskConf);
      } else {
        int label = maxSPtr - scoresPtr;
        if (std::find(detected_class.begin(), detected_class.end(),
                      class_labels_[label]) != detected_class.end()) {
          float x = *bboxesPtr++;
          float y = *bboxesPtr++;
          float w = *bboxesPtr++;
          float h = *bboxesPtr;
          float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, input_frame_w_);
          float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, input_frame_h_);
          float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, input_frame_w_);
          float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, input_frame_h_);
          cv::Rect_<float> bbox;
          bbox.x = x0;
          bbox.y = y0;
          bbox.width = x1 - x0;
          bbox.height = y1 - y0;

          cv::Mat maskConf = cv::Mat(1, num_seg_channels, CV_32F, maskConfsPtr);

          bboxes.push_back(bbox);
          labels.push_back(label);
          scores.push_back(score);
          maskConfs.push_back(maskConf);
        }
      }
    }
  }

  // Require OpenCV 4.7 for this function
  cv::dnn::NMSBoxesBatched(bboxes, scores, labels, params.obj_threshold,
                           params.nms_threshold, indices);
  std::cout<<bboxes.size()<<std::endl;
  std::cout<<labels.size()<<std::endl;
  std::cout<<indices.size()<<std::endl;

  // Obtain the segmentation masks
  cv::Mat masks;
  std::vector<Object> objs;
  int cnt = 0;
  for (auto &i : indices) {
    if (cnt >= params.num_detect) {
      break;
    }
    cv::Rect tmp = bboxes[i];
    Object obj;
    obj.label = labels[i];
    obj.rect = tmp;
    std::cout<<labels[i]<<std::endl;
    obj.probability = scores[i];
    masks.push_back(maskConfs[i]);
    objs.push_back(obj);
    cnt += 1;
  }

  // Convert segmentation mask to original frame
  if (!masks.empty()) {
    cv::Mat matmulRes = (masks * protos).t();
    std::cout<<masks.size()<<std::endl;
    std::cout<<indices.size()<<std::endl;
    std::cout<<indices.size()<<std::endl;

    cv::Mat maskMat = matmulRes.reshape(indices.size(), {SEG_W, SEG_H});

    std::vector<cv::Mat> maskChannels;
    cv::split(maskMat, maskChannels);

    cv::Rect roi;
    if (input_frame_h_ > input_frame_w_) {
      roi = cv::Rect(0, 0, SEG_W * input_frame_w_ / input_frame_h_, SEG_H);
    } else {
      roi = cv::Rect(0, 0, SEG_W, SEG_H * input_frame_h_ / input_frame_w_);
    }

    for (size_t i = 0; i < indices.size(); i++) {
      cv::Mat dest, mask;
      cv::exp(-maskChannels[i], dest);
      dest = 1.0 / (1.0 + dest);
      dest = dest(roi);
      cv::resize(dest, mask,
                 cv::Size(static_cast<int>(input_frame_w_),
                          static_cast<int>(input_frame_h_)),
                 cv::INTER_LINEAR);
      objs[i].box_mask = mask(objs[i].rect) > params.seg_threshold;
    }
  }

  return objs;
}

std::vector<Object> YoloV8::postprocessPose(
    std::unordered_map<std::string, std::vector<float>> &feature_vectors,
    const DetectionParams &params,
    const std::vector<std::string> &detected_class) {
  int num_anchors = m_trtEngine->getOutputInfo().at("output0").dims.d[2];
  int num_channels = m_trtEngine->getOutputInfo().at("output0").dims.d[1];
  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<int> indices;
  std::vector<std::vector<float>> kpss;

  cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F,
                           feature_vectors.at("output0").data());
  output = output.t();
  // Get all the YOLO proposals
  for (int i = 0; i < num_anchors; i++) {
    auto rowPtr = output.row(i).ptr<float>();
    auto bboxesPtr = rowPtr;
    auto scoresPtr = rowPtr + 4;
    auto kps_ptr = rowPtr + 5;
    float score = *scoresPtr;
    if (score > params.obj_threshold) {
      if (detected_class.empty() ||
          (detected_class.size() == 1 &&
           (detected_class[0] == "all" || detected_class[0] == ""))) {
        float x = *bboxesPtr++;
        float y = *bboxesPtr++;
        float w = *bboxesPtr++;
        float h = *bboxesPtr;

        float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, input_frame_w_);
        float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, input_frame_h_);
        float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, input_frame_w_);
        float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, input_frame_h_);

        cv::Rect_<float> bbox;
        bbox.x = x0;
        bbox.y = y0;
        bbox.width = x1 - x0;
        bbox.height = y1 - y0;

        std::vector<float> kps;
        for (int k = 0; k < num_kps_; k++) {
          float kpsX = *(kps_ptr + 3 * k) * m_ratio;
          float kpsY = *(kps_ptr + 3 * k + 1) * m_ratio;
          float kpsS = *(kps_ptr + 3 * k + 2);
          kpsX = std::clamp(kpsX, 0.f, input_frame_w_);
          kpsY = std::clamp(kpsY, 0.f, input_frame_h_);
          kps.push_back(kpsX);
          kps.push_back(kpsY);
          kps.push_back(kpsS);
        }

        bboxes.push_back(bbox);
        labels.push_back(0);  // All detected objects are people
        scores.push_back(score);
        kpss.push_back(kps);
      } else {
        int label = 0;
        if (std::find(detected_class.begin(), detected_class.end(),
                      class_labels_[label]) != detected_class.end()) {
          float x = *bboxesPtr++;
          float y = *bboxesPtr++;
          float w = *bboxesPtr++;
          float h = *bboxesPtr;

          float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, input_frame_w_);
          float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, input_frame_h_);
          float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, input_frame_w_);
          float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, input_frame_h_);

          cv::Rect_<float> bbox;
          bbox.x = x0;
          bbox.y = y0;
          bbox.width = x1 - x0;
          bbox.height = y1 - y0;

          std::vector<float> kps;
          for (int k = 0; k < num_kps_; k++) {
            float kpsX = *(kps_ptr + 3 * k) * m_ratio;
            float kpsY = *(kps_ptr + 3 * k + 1) * m_ratio;
            float kpsS = *(kps_ptr + 3 * k + 2);
            kpsX = std::clamp(kpsX, 0.f, input_frame_w_);
            kpsY = std::clamp(kpsY, 0.f, input_frame_h_);
            kps.push_back(kpsX);
            kps.push_back(kpsY);
            kps.push_back(kpsS);
          }

          bboxes.push_back(bbox);
          labels.push_back(label);  // All detected objects are people
          scores.push_back(score);
          kpss.push_back(kps);
        }
      }
    }
  }

  // Run NMS
  cv::dnn::NMSBoxesBatched(bboxes, scores, labels, params.obj_threshold,
                           params.nms_threshold, indices);

  std::vector<Object> objects;

  // Choose the top k detections
  int cnt = 0;
  for (auto &chosenIdx : indices) {
    if (cnt >= params.num_detect) {
      break;
    }

    Object obj{};
    obj.probability = scores[chosenIdx];
    obj.label = labels[chosenIdx];
    obj.rect = bboxes[chosenIdx];
    obj.kps = kpss[chosenIdx];
    objects.push_back(obj);

    cnt += 1;
  }

  return objects;
}

}  // namespace tensorrt_inference