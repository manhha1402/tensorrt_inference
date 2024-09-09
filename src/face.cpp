#include "tensorrt_inference/face.h"
namespace tensorrt_inference
{
Face::Face(const std::string& model_dir,const YAML::Node &config) : Model(model_dir,config) {
    if(config["obj_threshold"])
    {
        obj_threshold_ = config["obj_threshold"].as<float>();
    }
    if(config["nms_threshold"])
    {
        nms_threshold_ = config["nms_threshold"].as<float>();
    }
    if(config["landmark_std"])
    {
        landmark_std_ = config["landmark_std"].as<float>();
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
    std::cout<<"obj_threshold: "<<obj_threshold_<<std::endl;
    std::cout<<"nms_threshold: "<<nms_threshold_<<std::endl;
    std::cout<<"num_detect_: "<<num_detect_<<std::endl;
    std::cout<<"sub_vals: "<<sub_vals_[0]<<" "<<sub_vals_[1]<<" "<<sub_vals_[2]<<std::endl;
    std::cout<<"div_vals: "<<div_vals_[0]<<" "<<div_vals_[1]<<" "<<div_vals_[2]<<std::endl;

    std::cout<<"normalized: "<<normalized_<<std::endl;
    std::cout<<"swapBR: "<<swapBR_<<std::endl;


}
std::vector<FaceBox> Face::detectFaces(const cv::Mat &inputImageBGR)
{
  
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);
    // Call detectObjects with the GPU image
    return detectFaces(gpuImg);
}
std::vector<FaceBox> Face::detectFaces(cv::cuda::GpuMat &inputImageBGR)
{
    std::unordered_map<std::string, std::vector<float>> feature_vectors;
    doInference(inputImageBGR,feature_vectors);
    inputImageBGR.release();
    // Check if our model does only object detection or also supports segmentation
    std::vector<FaceBox> ret = postProcess(feature_vectors);
    return ret;
}
void Face::drawFaceLabels(cv::Mat &image, const std::vector<FaceBox> &faces, unsigned int scale)
{
   for (const auto &face : faces)
    {
        cv::Scalar color = cv::Scalar(255, 0, 0);
        cv::rectangle(image, face.rect, color, 2, cv::LINE_8, 0);
    }
}

}