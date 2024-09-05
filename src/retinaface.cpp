#include "tensorrt_inference/retinaface.h"

#include <opencv2/cudaimgproc.hpp>
namespace tensorrt_inference
{
RetinaFace::RetinaFace(const std::string& model_dir,const YAML::Node &config) : Face(model_dir,config)
{
}
static bool m_cmp(FaceBox a, FaceBox b) {
    if (a.score > b.score)
        return true;
    return false;
}
std::vector<FaceBox> RetinaFace::postProcess(std::vector<std::vector<std::vector<float>>> &feature_vectors)
{
    // std::cout<<1<<std::endl;
    
     const auto &outputDims = m_trtEngine->getOutputDims();
     const auto &inputDims = m_trtEngine->getInputDims();
    //  const std::vector<float>& bbox = feature_vectors[0][0];
    //  const std::vector<float>& conf = feature_vectors[0][1];
    //  const std::vector<float>& lmks = feature_vectors[0][2];

     float *bbox = &feature_vectors[0][0][0];
     float *conf = &feature_vectors[0][1][0];
     float *lmks = &feature_vectors[0][2][0];



    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];
    int m_INPUT_H = inputDims[0].d[1];
    int m_INPUT_W =  inputDims[0].d[2];
    std::vector<FaceBox> face_box;
    std::vector<anchorBox> anchor;

    create_anchor_retinaface(anchor, m_INPUT_W, m_INPUT_H);

    std::cout<<"ok2"<<std::endl;

    for (int i = 0; i < anchor.size(); ++i) {
        std::cout<<*(conf + 1)<<std::endl;
        if (*(conf + 1) * 100.0  > obj_threshold_) {
            anchorBox tmp = anchor[i];
            anchorBox tmp1;
            FaceBox result;
            //result.classId = 0;

            // decode bbox
            tmp1.cx = tmp.cx + *bbox * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(bbox + 1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(bbox + 2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(bbox + 3) * 0.2);

            // for (unsigned int i2=0; i2 < 5; i2++) {
            //     result.landmark[i2*2] = lround((tmp.cx + *(lmks + i2*2) * 0.1 * tmp.sx) * width);
            //     result.landmark[i2*2 + 1] = lround((tmp.cy + *(lmks + i2*2 + 1) * 0.1 * tmp.sy) * height);
            // }

            result.rect.x = (tmp1.cx - tmp1.sx / 2) * m_INPUT_W;
            result.rect.y = (tmp1.cy - tmp1.sy / 2) * m_INPUT_H;
            result.rect.width = (tmp1.cx + tmp1.sx / 2) * m_INPUT_W - result.rect.x;
            result.rect.height = (tmp1.cy + tmp1.sy / 2) * m_INPUT_H - result.rect.y;

            // Clip object box coordinates to network resolution
            result.rect.x = CLIP(result.rect.x, 0, m_INPUT_W - 1);
            result.rect.y = CLIP(result.rect.y, 0, m_INPUT_H - 1);
            result.rect.width = CLIP(result.rect.width, 0, m_INPUT_W - 1);
            result.rect.height = CLIP(result.rect.height, 0, m_INPUT_H - 1);


            result.score = *(conf + 1);

            //result.numLmks = NUM_LANDMARK;
  
            face_box.push_back(result);
            // printf("bbox: %f %f %f %f conf: %f\n", result.left, result.top, result.width, result.height, result.detectionConfidence);
        }
        bbox += 4;
        lmks += 10;
        conf += 2;
    }
    std::sort(face_box.begin(), face_box.end(), m_cmp);
    std::cout<<"num face before nms"<<face_box.size()<<std::endl;
     std::cout<<"nms_threshold_: "<<nms_threshold_<<std::endl;
     std::cout<<"obj_threshold_: "<<obj_threshold_<<std::endl;


    nms(face_box, nms_threshold_);
    if (face_box.size() > 100)
         face_box.resize(100);
    std::cout<<"num face"<<face_box.size()<<std::endl;
 
    return face_box;
}

void RetinaFace::nms(std::vector<FaceBox> &input_boxes, float NMS_THRESH) {
     std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).rect.width + 1) * (input_boxes.at(i).rect.height + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].rect.x, input_boxes[j].rect.x);
            float yy1 = std::max(input_boxes[i].rect.y, input_boxes[j].rect.y);
            float xx2 =
                std::min(input_boxes[i].rect.x + input_boxes[i].rect.width, input_boxes[j].rect.x + input_boxes[j].rect.width);
            float yy2 =
                std::min(input_boxes[i].rect.y + input_boxes[i].rect.height, input_boxes[j].rect.y + input_boxes[j].rect.height);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}



void RetinaFace::create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h) {
    anchor.clear();
    std::vector<std::vector<int>> feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (unsigned int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h / steps[i]));
        feature_map[i].push_back(ceil(w / steps[i]));
    }
    std::vector<int> minsize1 = {16, 32};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {64, 128};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {256, 512};
    min_sizes[2] = minsize3;

    for (unsigned int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (unsigned int l = 0; l < min_size.size(); ++l) {
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