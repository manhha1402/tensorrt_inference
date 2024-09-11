

#include "tensorrt_inference/detection.h"
namespace tensorrt_inference
{
Detection::Detection(const std::string& model_dir,const YAML::Node &config) : Model(model_dir,config) {
    
    
   
    if(config["num_kps"])
    {
        num_kps_ = config["num_kps"].as<int>();
    }
    std::string labels_file = model_dir + "/" + config["labels_file"].as<std::string>();
    if (Util::doesFileExist(std::filesystem::path(labels_file)))
    {
        class_labels_ = readClassLabel(labels_file);
    }
    else {
        spdlog::error("label file is not existed!");
    }
    CATEGORY = class_labels_.size();
    class_colors_.resize(CATEGORY);
    srand((int) time(nullptr));
    for (cv::Scalar &class_color : class_colors_)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    std::cout<<"sub_vals: "<<sub_vals_[0]<<" "<<sub_vals_[1]<<" "<<sub_vals_[2]<<std::endl;
    std::cout<<"div_vals: "<<div_vals_[0]<<" "<<div_vals_[1]<<" "<<div_vals_[2]<<std::endl;
    std::cout<<"normalized: "<<normalized_<<std::endl;
    std::cout<<"swapBR: "<<swapBR_<<std::endl;

}

std::vector<Object> Detection::detectObjects(const cv::Mat &inputImageBGR, const DetectionParams& params)
{
 // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);
    // Call detectObjects with the GPU image
    return detectObjects(gpuImg,params);
}

std::vector<Object> Detection::detectObjects(cv::cuda::GpuMat &inputImageBGR, const DetectionParams& params)
{
    std::unordered_map<std::string, std::vector<float>> feature_vectors;
    doInference(inputImageBGR,feature_vectors);
    inputImageBGR.release();
    // Check if our model does only object detection or also supports segmentation
    std::vector<Object> ret= postprocess(feature_vectors,params);
    return ret;
}

void Detection::drawBBoxLabel(cv::Mat &image, const Object &object, const DetectionParams& params, unsigned int scale)
{
// Choose the color
    int colorIndex = object.label % class_colors_.size(); // We have only defined 80 unique colors
    cv::Scalar color = cv::Scalar(class_colors_[colorIndex][0], class_colors_[colorIndex][1], class_colors_[colorIndex][2]);
    float meanColor = cv::mean(color)[0];
    cv::Scalar txtColor;
    if (meanColor > 0.5) {
        txtColor = cv::Scalar(0, 0, 0);
    } else {
        txtColor = cv::Scalar(255, 255, 255);
    }
    const auto &rect = object.rect;
    // Draw rectangles and text
    char text[256];
    sprintf(text, "%s %.1f%%", class_labels_[object.label].c_str(), object.probability * 100);

    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7 * 255;

    int x = object.rect.x;
    int y = object.rect.y + 1;

    cv::rectangle(image, rect, color * 255, scale + 1);

    cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), txt_bk_color, -1);

    cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

    // Pose estimation
    if (!object.kps.empty()) {
        auto &kps = object.kps;
        for (int k = 0; k < num_kps_ + 2; k++) {
            if (k < num_kps_) {
                int kpsX = std::round(kps[k * 3]);
                int kpsY = std::round(kps[k * 3 + 1]);
                float kpsS = kps[k * 3 + 2];
                if (kpsS > params.kps_threshold) {
                    cv::Scalar kpsColor = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
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

            if (pos1S > params.kps_threshold && pos2S > params.kps_threshold) {
                cv::Scalar limbColor = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
            }
        }
    }
}
void Detection::drawSegmentation(cv::Mat &mask, const Object &object)
{
// Choose the color
    int colorIndex = object.label % class_colors_.size(); // We have only defined 80 unique colors
    // Add the mask for said object
    mask(object.rect).setTo(class_colors_[colorIndex] * 255, object.box_mask);
}

void Detection::drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects,const DetectionParams& params,const std::vector<std::string>& detected_class, unsigned int scale)
{
    // If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].box_mask.empty()) 
    {
        cv::Mat mask = image.clone();
        for (const auto &object : objects) {
            if(detected_class.empty())
            {
               drawSegmentation(mask,object);
            }
            else
            {
               if(std::find(detected_class.begin(), detected_class.end(), class_labels_[object.label])!= detected_class.end())
               {
                 drawSegmentation(mask,object);
               }
            }
        }
        // Add all the masks to our image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    // Bounding boxes and annotations
    for (auto &object : objects) {
        if(detected_class.empty())
        {
            drawBBoxLabel(image,object,params,scale);
        }
        else
        {
            if(std::find(detected_class.begin(), detected_class.end(), class_labels_[object.label])!= detected_class.end())
            {
                drawBBoxLabel(image,object,params,scale);
            }
        }
    }
}
}