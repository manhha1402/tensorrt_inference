#include "tensorrt_inference/retinaface.h"

#include <opencv2/cudaimgproc.hpp>
namespace tensorrt_inference
{
RetinaFace::RetinaFace(const std::string& model_dir,const YAML::Node &config) : Face(model_dir,config)
{
}

std::vector<FaceBox> RetinaFace::postProcess(std::unordered_map<std::string, std::vector<float>> &feature_vectors)
{    
    const auto &input_info = m_trtEngine->getInputInfo().begin();
    const std::vector<float>& conf = feature_vectors["593"] ;
    const std::vector<float>& bbox = feature_vectors["output0"];
    const std::vector<float>& lmks = feature_vectors["592"];
    std::vector<anchorBox> anchor;
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    create_anchor_retinaface(anchor, input_info->second.dims.d[3], input_info->second.dims.d[2]);
    for (int i = 0; i < anchor.size(); ++i) {
        if (conf[i * 2 + 1] > obj_threshold_) 
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
            float width = ((tmp1.cx + tmp1.sx / 2) * input_info->second.dims.d[3] - x);
            float height = ((tmp1.cy + tmp1.sy / 2) * input_info->second.dims.d[2] - y);

            rect.x = std::clamp(x * m_ratio, 0.f, input_frame_w_);
            rect.y = std::clamp(y * m_ratio, 0.f, input_frame_h_);
            rect.width = width * m_ratio;
            rect.height = height * m_ratio;
            bboxes.push_back(rect);
            scores.push_back(conf[i * 2 + 1]);
            labels.push_back(1);
        }

    }
   cv::dnn::NMSBoxesBatched(bboxes, scores, labels, obj_threshold_, nms_threshold_, indices);
    // Choose the top k detections
    std::vector<FaceBox> num_faces;
    int cnt = 0;
    for (auto &chosenIdx : indices)
    {
        if (cnt >= num_detect_)
        {
            break;
        }
        FaceBox face_box;
        face_box.score = scores[chosenIdx];
        face_box.rect = bboxes[chosenIdx];
        num_faces.push_back(face_box);
        cnt += 1;
    }
    return num_faces;
}



void RetinaFace::create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h) {
    
    anchor.clear();
    std::vector<std::vector<int>> feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i)
    {
        feature_map[i].push_back(ceil(h / steps[i]));
        feature_map[i].push_back(ceil(w / steps[i]));
    }
    std::vector<int> minsize1 = {10, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 64};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {128, 256};
    min_sizes[2] = minsize3;

    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
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

}