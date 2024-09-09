#include "tensorrt_inference/yolov9.h"
namespace tensorrt_inference
{
YoloV9::YoloV9(const std::string& model_dir,const YAML::Node &config) : 
Detection(model_dir,config)
{}
std::vector<Object> YoloV9::postprocess(std::unordered_map<std::string, std::vector<float>> &feature_vectors)
{
    
    if(m_trtEngine->getOutputInfo().size () == 1)
    {
        return postprocessDetect(feature_vectors);
    }
    else if (m_trtEngine->getOutputInfo().size () == 2)
    {
        return postProcessSegmentation(feature_vectors);
    }
    else{

        return std::vector<Object>();
    }
 
}

    // Postprocess the output
std::vector<Object> YoloV9::postprocessDetect(std::unordered_map<std::string, std::vector<float>> &feature_vectors)
{ 
    int num_anchors = m_trtEngine->getOutputInfo().at("output0").dims.d[2];
    int num_channels = m_trtEngine->getOutputInfo().at("output0").dims.d[1];

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    /*
    cv::Mat res_mat = cv::Mat(CATEGORY + 4, num_anchors, CV_32F, feature_vector["output0"].data());
    res_mat = res_mat.t();
    cv::Mat prob_mat;
    cv::reduce(res_mat.colRange(4, CATEGORY + 4), prob_mat, 1, cv::REDUCE_MAX);
    float *out = res_mat.ptr<float>(0);
    // Get all the YOLO proposals
    for (int i = 0; i < num_anchors; i++) {
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
    */
    // Get all the YOLO proposals
    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, feature_vectors.at("output0").data());
    output = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + CATEGORY);
        float score = *maxSPtr;
        if (score > obj_threshold_) {
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

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
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

    // Postprocess the output for segmentation model
std::vector<Object> YoloV9::postProcessSegmentation(std::unordered_map<std::string, std::vector<float>> &feature_vectors)
{

    const auto& output0_info = m_trtEngine->getOutputInfo().at("output0");
    const auto& output1_info = m_trtEngine->getOutputInfo().at("output1");

    const int num_net0_channels = output0_info.dims.d[1];
    const int num_seg_channels = output1_info.dims.d[1];

    const int num_net0_anchors = output0_info.dims.d[2];

    const int SEG_H = output1_info.dims.d[2];
    const int SEG_W = output1_info.dims.d[3];

    const auto numClasses = num_net0_channels - num_seg_channels - 4;

    

    cv::Mat output = cv::Mat(num_net0_channels, num_net0_anchors, CV_32F, feature_vectors.at("output0").data());
    output = output.t();

    cv::Mat protos = cv::Mat(num_seg_channels, SEG_H * SEG_W, CV_32F, feature_vectors.at("output1").data());

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
        if (score > obj_threshold_) {
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
        }
    }

    // Require OpenCV 4.7 for this function
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, obj_threshold_, nms_threshold_, indices);

    // Obtain the segmentation masks
    cv::Mat masks;
    std::vector<Object> objs;
    int cnt = 0;
    for (auto &i : indices) {
        if (cnt >= num_detect_) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.probability = scores[i];
        masks.push_back(maskConfs[i]);
        objs.push_back(obj);
        cnt += 1;
    }

    // Convert segmentation mask to original frame
    if (!masks.empty()) {
        cv::Mat matmulRes = (masks * protos).t();
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
            cv::resize(dest, mask, cv::Size(static_cast<int>(input_frame_w_), static_cast<int>(input_frame_h_)), cv::INTER_LINEAR);
            objs[i].box_mask = mask(objs[i].rect) > seg_threshold_;
        }
    }

    return objs;
}


}