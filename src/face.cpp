#include "tensorrt_inference/face.h"
namespace tensorrt_inference
{
Face::Face(const std::string& model_dir,const YAML::Node &config) : Model(model_dir,config) {
    if(config["landmark_std"])
    {
        landmark_std_ = config["landmark_std"].as<float>();
    }
    std::cout<<"sub_vals: "<<sub_vals_[0]<<" "<<sub_vals_[1]<<" "<<sub_vals_[2]<<std::endl;
    std::cout<<"div_vals: "<<div_vals_[0]<<" "<<div_vals_[1]<<" "<<div_vals_[2]<<std::endl;
    std::cout<<"normalized: "<<normalized_<<std::endl;
    std::cout<<"swapBR: "<<swapBR_<<std::endl;


}
std::vector<FaceBox> Face::detectFaces(const cv::Mat &inputImageBGR,const ModelParams& params)
{
  
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);
    // Call detectObjects with the GPU image
    return detectFaces(gpuImg,params);
}
std::vector<FaceBox> Face::detectFaces(cv::cuda::GpuMat &inputImageBGR,const ModelParams& params)
{
    std::unordered_map<std::string, std::vector<float>> feature_vectors;
    doInference(inputImageBGR,feature_vectors);
    inputImageBGR.release();
    // Check if our model does only object detection or also supports segmentation
    std::vector<FaceBox> ret = postProcess(feature_vectors,params);
    return ret;
}
cv::Mat Face::drawFaceLabels(const cv::Mat &image, const std::vector<FaceBox> &faces, unsigned int scale)
{
    cv::Mat result = image.clone();
   for (const auto &face : faces)
    {
        cv::Scalar color = cv::Scalar(255, 0, 0);
        cv::rectangle(result, face.rect, color, scale, cv::LINE_8, 0);
    }
    return result;
}

}