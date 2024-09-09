

#include "tensorrt_inference/detection.h"
namespace tensorrt_inference
{
Detection::Detection(const std::string& model_dir,const YAML::Node &config) : Model(model_dir,config) {
    
    if(config["obj_threshold"])
    {
        obj_threshold_ = config["obj_threshold"].as<float>();
    }
    if(config["nms_threshold"])
    {
        nms_threshold_ = config["nms_threshold"].as<float>();
    }
    if(config["seg_threshold"])
    {
        seg_threshold_ = config["seg_threshold"].as<float>();
    }
    if(config["num_detect"])
    {
        num_detect_ = config["num_detect"].as<int>();
    }
    if(config["normalized"])
    {
        normalized_ = config["normalized"].as<bool>();
    }
     if(config["swapBR"])
    {
        swapBR_ = config["swapBR"].as<bool>();
    }
    if(config["sub_vals"])
    {
       sub_vals_ = config["sub_vals"].as<std::vector<float>>();
    }
    if(config["div_vals"])
    {
        div_vals_ = config["div_vals"].as<std::vector<float>>();
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


    std::cout<<"obj_threshold: "<<obj_threshold_<<std::endl;
    std::cout<<"nms_threshold: "<<nms_threshold_<<std::endl;
    std::cout<<"num_detect: "<<num_detect_<<std::endl;

    std::cout<<"sub_vals: "<<sub_vals_[0]<<" "<<sub_vals_[1]<<" "<<sub_vals_[2]<<std::endl;
    std::cout<<"div_vals: "<<div_vals_[0]<<" "<<div_vals_[1]<<" "<<div_vals_[2]<<std::endl;
    std::cout<<"normalized: "<<normalized_<<std::endl;
    std::cout<<"swapBR: "<<swapBR_<<std::endl;

}

std::vector<Object> Detection::detectObjects(const cv::Mat &inputImageBGR)
{
 // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);
    // Call detectObjects with the GPU image
    return detectObjects(gpuImg);
}

std::vector<Object> Detection::detectObjects(cv::cuda::GpuMat &inputImageBGR)
{
    std::unordered_map<std::string, std::vector<float>> feature_vectors;
    doInference(inputImageBGR,feature_vectors);
    inputImageBGR.release();
    // Check if our model does only object detection or also supports segmentation
    std::vector<Object> ret= postprocess(feature_vectors);
    return ret;
}

void Detection::drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale)
{
     // If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].box_mask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto &object : objects) {
            // Choose the color
            int colorIndex = object.label % class_colors_.size(); // We have only defined 80 unique colors
            // Add the mask for said object
            mask(object.rect).setTo(class_colors_[colorIndex] * 255, object.box_mask);
        }
        // Add all the masks to our image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }
    
    for (auto &object : objects) {
    int colorIndex = object.label % class_colors_.size(); 
    float meanColor = cv::mean(class_colors_[colorIndex])[0];
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

    cv::Scalar txt_bk_color = class_colors_[colorIndex] * 0.7 * 255;

    int x = object.rect.x;
    int y = object.rect.y + 1;

    cv::rectangle(image, rect, class_colors_[colorIndex] * 255, scale + 1);

    cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), txt_bk_color, -1);

    cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);
    }
}

    
}