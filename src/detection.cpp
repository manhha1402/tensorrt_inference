

#include "tensorrt_inference/detection.h"
namespace tensorrt_inference
{
Detection::Detection(const std::string& model_dir,const YAML::Node &config) : Model(model_dir,config) {
    std::string labels_file = model_dir + "/" + config["labels_file"].as<std::string>();
    obj_threshold_ = config["obj_threshold"].as<float>();
    nms_threshold_ = config["nms_threshold"].as<float>();
    agnostic_ = config["agnostic"].as<bool>();
    strides_ = config["strides"].as<std::vector<int>>();
    num_anchors_ = config["num_anchors"].as<std::vector<int>>();
    int index = 0;
    const auto &inputDims = m_trtEngine->getInputDims();

    for (const int &stride : strides_)
    {
        int num_anchor = num_anchors_[index] !=0 ? num_anchors_[index] : 1;
        num_rows_ += int(inputDims[0].d[1] / stride) * int(inputDims[0].d[2] / stride) * num_anchor;
        index+=1;
    }
    std::cout<<"chanels: "<<inputDims[0].d[0]<<std::endl;
    std::cout<<"height: "<<inputDims[0].d[1]<<std::endl;
    std::cout<<"width: "<<inputDims[0].d[2]<<std::endl;
    std::cout<<"num_rows: "<<num_rows_<<std::endl;
    std::cout<<"strides: "<<strides_.size()<<std::endl;
    std::cout<<"num_anchors: "<<num_anchors_.size()<<std::endl;

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
}

std::vector<Object> Detection::detectObjects(const cv::Mat &inputImageBGR)
{
 // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);
    // Call detectObjects with the GPU image
    return detectObjects(gpuImg);
}

std::vector<Object> Detection::detectObjects(const cv::cuda::GpuMat &inputImageBGR)
{
    std::vector<std::vector<std::vector<float>>> feature_vectors;
    doInference(inputImageBGR,feature_vectors);
    // Check if our model does only object detection or also supports segmentation
    std::vector<Object> ret;
    const auto &numOutputs = m_trtEngine->getOutputDims().size();
    if (numOutputs == 1) {
        // Object detection or pose estimation
        // Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
        std::vector<float> feature_vector;
        Engine<float>::transformOutput(feature_vectors, feature_vector);
        // Object detection
        ret = postprocessDetect(feature_vector);
    } else {
        throw std::runtime_error("Incorrect number of outputs!");
    }

    return ret;
}

void Detection::drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale)
{
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