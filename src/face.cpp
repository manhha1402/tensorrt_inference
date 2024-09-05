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
void Face::preprocess(const cv::Mat& img) {
    const auto &inputDims = m_trtEngine->getInputDims();
    m_scale_h = (float)inputDims[0].d[1] / img.cols;
    m_scale_w = (float)inputDims[0].d[2] / img.rows;
  // Release input vector
    m_input.release();

    // Resize
    int w, h, x, y;
    if (m_scale_h > m_scale_w)
    {
        w = inputDims[0].d[2];
        h = m_scale_w * img.rows;
        x = 0;
        y = (inputDims[0].d[1] - h) / 2;
    }
    else
    {
        w = m_scale_h * img.cols;
        h = inputDims[0].d[1];
        x = (inputDims[0].d[2] - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(inputDims[0].d[1], inputDims[0].d[2], CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    // Normalize
    out.convertTo(out, CV_32F);
    out = out - cv::Scalar(104, 117, 123);
    std::vector<cv::Mat> temp;
    cv::split(out, temp);
    for (int i = 0; i < temp.size(); i++)
    {
        m_input.push_back(temp[i]);
    }
}

std::vector<FaceBox> Face::detectFaces(const cv::Mat &inputImageBGR)
{
    //preprocess(inputImageBGR);
    // m_frameHeight = inputImageBGR.rows;
    // m_frameWidth = inputImageBGR.cols;
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
    std::cout<<"num output: "<<numOutputs<<std::endl;
    std::cout<<"feature_vectors size: "<<feature_vectors.size()<<std::endl;
    // for(const auto features: feature_vectors)
    // {
    //     std::cout<<"features size: "<<features.size()<<std::endl;
    // }

    //if (numOutputs == 1) {
        // Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
    //std::vector<float> feature_vector;
   //Engine<float>::transformOutput(feature_vectors, feature_vector);
    //std::cout<<"num size"<<feature_vector.size()<<std::endl;
    // Object detection
    ret = postProcess(feature_vectors);
    // } else {
    //     throw std::runtime_error("Incorrect number of outputs!");
    // }

    return ret;
}
void Face::drawFaceLabels(cv::Mat &image, const std::vector<FaceBox> &faces, unsigned int scale)
{

}

}