#include "tensorrt_inference/retinaface.h"

#include <opencv2/cudaimgproc.hpp>
namespace tensorrt_inference
{
  RetinaFace::RetinaFace(const std::string &model_name,
                         tensorrt_inference::Options options,
                         const std::filesystem::path &model_dir)
      : Detection(model_name, options, model_dir) {}

  std::vector<Object> RetinaFace::postprocess(
      std::unordered_map<std::string, std::vector<float>> &feature_vectors,
      const DetectionParams &params,
      const std::vector<std::string> &detected_class)
  {
    const auto &input_info = m_trtEngine->getInputInfo().begin();

    std::vector<float> conf, bbox, lmks;
    for (auto it = feature_vectors.begin(); it != feature_vectors.end(); ++it)
    {
      std::cout<<it->first<<std::endl;
      if(it->first=="face_rpn_cls_prob_reshape_stride16")
       {
          for (const auto &val : it->second)
        {
          std::cout << val << std::endl;
        }
       }
      // const auto &output_info = m_trtEngine->getOutputInfo().at(it->first);

      // int rows = output_info.dims.d[2];
      // int dimensions = output_info.dims.d[1];

      // cv::Mat outputs(dimensions, rows, CV_32F, it->second.data());
      // outputs = outputs.reshape(1, dimensions);
      // cv::transpose(outputs, outputs);
      // std::vector<float> output_data(outputs.rows * outputs.cols);
      // std::memcpy(output_data.data(), outputs.ptr<float>(), output_data.size() * sizeof(float));
      // if (output_data.size() == 33600) //
      // {
      //   for (const auto &val : it->second)
      //   {
      //     std::cout << val << std::endl;
      //   }
      //   conf = output_data;
      // }
      // else if (output_data.size() == 67200)
      // {
      //   bbox = output_data;
      // }
      // else if(it.second.size() == 168000)
      // {
      //     lmks = it.second;
      // }
    }
    std::vector<anchorBox> anchor;
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    // std::vector<std::vector<cv::Point2f>> face_lmks;
    create_anchor_retinaface(anchor, input_info->second.dims.d[3],
                             input_info->second.dims.d[2]);
    for (size_t i = 0; i < anchor.size(); ++i)
    {

      if (conf[i * 2 + 1] > params.obj_threshold)
      {

        anchorBox tmp = anchor[i];
        anchorBox tmp1;
        cv::Rect rect;
        // decode bbox
        tmp1.cx = tmp.cx + bbox[i * 4 + 0] * 0.1 * tmp.sx;
        tmp1.cy = tmp.cy + bbox[i * 4 + 1] * 0.1 * tmp.sy;
        tmp1.sx = tmp.sx * exp(bbox[i * 4 + 2] * 0.2);
        tmp1.sy = tmp.sy * exp(bbox[i * 4 + 3] * 0.2);

        float x = (tmp1.cx - tmp1.sx / 2) * input_info->second.dims.d[3];
        float y = (tmp1.cy - tmp1.sy / 2) * input_info->second.dims.d[2];
        float width =
            ((tmp1.cx + tmp1.sx / 2) * input_info->second.dims.d[3] - x);
        float height =
            ((tmp1.cy + tmp1.sy / 2) * input_info->second.dims.d[2] - y);
        // std::vector<cv::Point2f> kps;
        //  for (unsigned int i2 = 0; i2 < 5; i2++) {
        //     cv::Point2f lmk;
        //     lmk.x = (tmp.cx + lmks[i2 * 2] * 0.1 * tmp.sx) *
        //     input_info->second.dims.d[3]; lmk.y = (tmp.cy + lmks[i2 * 2 + 1] *
        //     0.1 * tmp.sy) * input_info->second.dims.d[2]; lmk.x =
        //     std::clamp(lmk.x * m_ratio, 0.f, input_frame_w_); lmk.y =
        //     std::clamp(lmk.y * m_ratio, 0.f, input_frame_h_);
        //     kps.push_back(lmk);
        // }
        rect.x = std::clamp(x * ratios_[0], 0.f, input_frame_w_);
        rect.y = std::clamp(y * ratios_[1], 0.f, input_frame_h_);
        rect.width = width * ratios_[0];
        rect.height = height * ratios_[1];
        if (0 <= rect.x && 0 <= rect.width &&
            rect.x + rect.width <= input_frame_w_ && 0 <= rect.y &&
            0 <= rect.height && rect.y + rect.height <= input_frame_h_)
        {

          bboxes.push_back(rect);
          scores.push_back(conf[i * 2 + 1]);
          // face_lmks.push_back(kps);
          labels.push_back(1);
        }
      }
    }
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, params.obj_threshold,
                             params.nms_threshold, indices);

    // Choose the top k detections
    std::vector<Object> num_faces;
    int cnt = 0;
    for (auto &chosenIdx : indices)
    {
      if (cnt >= params.num_detect)
      {
        break;
      }
      Object face_box;
      face_box.probability = scores[chosenIdx];
      face_box.rect = bboxes[chosenIdx];
      face_box.label = 0;
      // face_box.landmarks = face_lmks[chosenIdx];
      num_faces.push_back(face_box);
      cnt += 1;
    }
    return num_faces;
  }

  void RetinaFace::create_anchor_retinaface(std::vector<anchorBox> &anchor, int w,
                                            int h)
  {
    anchor.clear();
    std::vector<std::vector<int>> feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (unsigned int i = 0; i < feature_map.size(); ++i)
    {
      feature_map[i].push_back(ceil(h / steps[i]));
      feature_map[i].push_back(ceil(w / steps[i]));
    }
    std::vector<int> minsize1 = {16, 32};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {64, 128};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {256, 512};
    min_sizes[2] = minsize3;

    for (unsigned int k = 0; k < feature_map.size(); ++k)
    {
      std::vector<int> min_size = min_sizes[k];
      for (int i = 0; i < feature_map[k][0]; ++i)
      {
        for (int j = 0; j < feature_map[k][1]; ++j)
        {
          for (unsigned int l = 0; l < min_size.size(); ++l)
          {
            float s_kx = min_size[l] * 1.0 / w;
            float s_ky = min_size[l] * 1.0 / h;
            float cx = (j + 0.5) * steps[k] / w;
            float cy = (i + 0.5) * steps[k] / h;
            anchorBox axil = {cx, cy, s_kx, s_ky};
            anchor.push_back(axil);
          }
        }
      }
    }
  }

} // namespace tensorrt_inference