#include "tensorrt_inference/retinaface.h"
#include <opencv2/cudaimgproc.hpp>
namespace tensorrt_inference
{
RetinaFace::RetinaFace(const std::string& model_dir,const YAML::Node &config) : Face(model_dir,config)
{
}

std::vector<FaceBox> RetinaFace::postProcess(std::vector<float> &feature_vector)
{
    const auto &outputDims = m_trtEngine->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];
    std::vector<FaceBox> face_box;
    return face_box;
}
void RetinaFace::generateAnchors() {
    float base_cx = 7.5;
    float base_cy = 7.5;
    refer_matrix = cv::Mat(sum_of_feature_, bbox_head_, CV_32FC1);
    int line = 0;
    for(size_t feature_map = 0; feature_map < feature_maps_.size(); feature_map++) {
        for (int height = 0; height < feature_maps_[feature_map][0]; ++height) {
            for (int width = 0; width < feature_maps_[feature_map][1]; ++width) {
                for (int anchor = 0; anchor < (int)anchor_sizes[feature_map].size(); ++anchor) {
                    auto *row = refer_matrix.ptr<float>(line);
                    row[0] = base_cx + (float)width * feature_steps_[feature_map];
                    row[1] = base_cy + (float)height * feature_steps_[feature_map];
                    row[2] = anchor_sizes[feature_map][anchor];
                    line++;
                }
            }
        }
    }
}
}