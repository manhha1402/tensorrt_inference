#include "tensorrt_inference/yolov9.h"
namespace tensorrt_inference
{
YoloV9::YoloV9(const std::string& model_dir,const YAML::Node &config) : 
Detection(model_dir,config)
{}
std::vector<Object> YoloV9::postprocessDetect(std::unordered_map<std::string, std::vector<float>> &feature_vector)
{
    auto numAnchors = m_trtEngine->getOutputInfo().at("output0").dims.d[2];
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    cv::Mat res_mat = cv::Mat(CATEGORY + 4, numAnchors, CV_32F, feature_vector["output0"].data());
    res_mat = res_mat.t();
    cv::Mat prob_mat;
    cv::reduce(res_mat.colRange(4, CATEGORY + 4), prob_mat, 1, cv::REDUCE_MAX);
    float *out = res_mat.ptr<float>(0);
    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) {
        float *row = out + i * (CATEGORY + 4);
        float prob = *prob_mat.ptr<float>(i);
        if (prob < obj_threshold_)
            continue;
        auto max_pos = std::max_element(row + 4, row + CATEGORY + 4);
        int label = max_pos - row - 4;
        float x = row[0] ;
        float y = row[1] ;
        float w = row[2] ;
        float h = row[3] ;
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
        scores.push_back(prob);
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, obj_threshold_, nms_threshold_, indices);

    std::vector<Object> objects;
    // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices) {
        if (cnt >= num_detect_) {
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
}