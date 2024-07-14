#pragma once
#include "tensorrt_inference/utils.h"
#include "tensorrt_inference/tensorrt_api/engine.h"
#include <fstream>
namespace tensorrt_inference
{
// Config the behavior of the YoloV8 detector.
// Can pass these arguments as command line parameters.
struct YoloV8Config {
    // The precision to be used for inference
    Precision precision = Precision::FP16;
    // Calibration data directory. Must be specified when using INT8 precision.
    std::string calibrationDataDirectory;
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.65f;
    // Max number of detected objects to return
    int topK = 100;
    // Segmentation config options
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // Pose estimation options
    int numKPS = 17;
    float kpsThreshold = 0.5f;
    // Class thresholds (default are COCO classes)
    // std::vector<std::string> classNames = {
    //     "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    //     "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    //     "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    //     "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    //     "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    //     "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    //     "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    //     "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    //     "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    //     "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    //     "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    //     "teddy bear",     "hair drier", "toothbrush"};
      std::vector<std::string> classNames = {"defect"};
};

class YoloV8 {
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    YoloV8(const std::string &onnxModelPath, const YoloV8Config &config);

    // Detect the objects in the image
    std::vector<Object> detectObjects(const cv::Mat &inputImageBGR);
    std::vector<Object> detectObjects(const cv::cuda::GpuMat &inputImageBGR);

    // Draw the object bounding boxes and labels on the image
    void drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale = 2);

private:
    // Preprocess the input
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg);

    // Postprocess the output
    std::vector<Object> postprocessDetect(std::vector<float> &featureVector);

    // Postprocess the output for pose model
    std::vector<Object> postprocessPose(std::vector<float> &featureVector);

    // Postprocess the output for segmentation model
    std::vector<Object> postProcessSegmentation(std::vector<std::vector<float>> &featureVectors);

    std::unique_ptr<Engine<float>> m_trtEngine = nullptr;

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

    // Filter thresholds
    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    // Segmentation constants
    const int SEG_CHANNELS;
    const int SEG_H;
    const int SEG_W;
    const float SEGMENTATION_THRESHOLD;

    // Object classes as strings
    const std::vector<std::string> CLASS_NAMES;

    // Pose estimation constant
    const int NUM_KPS;
    const float KPS_THRESHOLD;

 

    const std::vector<std::vector<unsigned int>> KPS_COLORS = {
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},   {255, 128, 0},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51, 153, 255},
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};

    const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
                                                             {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
                                                             {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};
};
}