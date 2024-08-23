#include "tensorrt_inference/face.h"
namespace tensorrt_inference
{
Face::Face(const std::string& model_dir,const YAML::Node &config) : Model(model_dir,config) {
    obj_threshold_ = config["obj_threshold"].as<float>();
    nms_threshold_ = config["nms_threshold"].as<float>();
    detect_mask_ = config["detect_mask"].as<bool>();
    mask_thresh_ = config["mask_thresh"].as<float>();
    landmark_std_ = config["landmark_std"].as<float>();
    feature_steps_ = config["feature_steps"].as<std::vector<int>>();
    const auto &inputDims = m_trtEngine->getInputDims();
    for (const int step: feature_steps_) {
        assert(step != 0);
        int feature_height = inputDims[0].d[1] / step;
        int feature_width = inputDims[0].d[2] / step;
        std::vector<int> feature_map = { feature_height, feature_width };
        feature_maps_.push_back(feature_map);
        int feature_size = feature_height * feature_width;
        feature_sizes_.push_back(feature_size);
    }
}

std::vector<FaceBox> Face::detectFaces(const cv::Mat &inputImageBGR)
{
 // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);
    // Call detectObjects with the GPU image
    return detectFaces(gpuImg);
}
std::vector<FaceBox> Face::detectFaces(const cv::cuda::GpuMat &inputImageBGR)
{
    std::vector<std::vector<std::vector<float>>> feature_vectors;
    doInference(inputImageBGR,feature_vectors);
    // Check if our model does only object detection or also supports segmentation
    std::vector<FaceBox> ret;
    const auto &numOutputs = m_trtEngine->getOutputDims().size();
    std::cout<<"num output"<<numOutputs<<std::endl;
    std::cout<<"feature_vectors size"<<feature_vectors.size()<<std::endl;
    std::cout<<"feature_vectors[0] size"<<feature_vectors[0].size()<<std::endl;

    //if (numOutputs == 1) {
        // Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
    std::vector<float> feature_vector;
    Engine<float>::transformOutput(feature_vectors, feature_vector);
    std::cout<<"num size"<<feature_vector.size()<<std::endl;
    // Object detection
    ret = postProcess(feature_vector);
    // } else {
    //     throw std::runtime_error("Incorrect number of outputs!");
    // }

    return ret;
}
void Face::drawFaceLabels(cv::Mat &image, const std::vector<FaceBox> &faces, unsigned int scale)
{

}

}