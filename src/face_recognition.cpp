#include "tensorrt_inference/face_recognition.h"

#include <opencv2/cudaimgproc.hpp>
namespace tensorrt_inference
{
    FaceRecognition::FaceRecognition(const std::string &model_name,
                                     tensorrt_inference::Options options,
                                     const std::filesystem::path &model_dir)
        : Model(model_name, options, model_dir) {}

} // namespace tensorrt_inference