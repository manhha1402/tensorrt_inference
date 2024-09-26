#include "tensorrt_inference/arcface.h"

#include <opencv2/cudaimgproc.hpp>
namespace tensorrt_inference {
ArcFace::ArcFace(const std::string &model_name,
                 tensorrt_inference::Options options,
                 const std::filesystem::path &model_dir)
    : Detection(model_name, options, model_dir) {}
std::vector<CroppedFace> ArcFace::getCroppedFaces(
    const cv::Mat &frame, const std::vector<Object> &faces) {
  const auto &input_info = m_trtEngine->getInputInfo().begin();
  std::vector<CroppedFace> cropped_faces;
  int i = 0;
  for (auto &face : faces) {
    cv::Mat tempCrop = frame(face.rect);
    CroppedFace currFace;
    cv::resize(
        tempCrop, currFace.face_mat,
        cv::Size(input_info->second.dims.d[2], input_info->second.dims.d[3]), 0,
        0, cv::INTER_CUBIC);  // resize to network dimension input
    currFace.face = currFace.face_mat.clone();
    currFace.rect = face.rect;
    cropped_faces.push_back(currFace);
    i++;
  }
  return cropped_faces;
}

void ArcFace::preprocessFace(cv::Mat &face, cv::Mat &output) {
  cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
  face.convertTo(face, CV_32F);
  face = (face - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
  std::vector<cv::Mat> temp;
  cv::split(face, temp);
  for (int i = 0; i < temp.size(); i++) {
    output.push_back(temp[i]);
  }
}

// void ArcFace::preprocessFaces() {
//     for (int i = 0; i < croppedFaces.size(); i++) {
//         cv::cvtColor(croppedFaces[i].faceMat, croppedFaces[i].faceMat,
//         cv::COLOR_BGR2RGB);
//         croppedFaces[i].faceMat.convertTo(croppedFaces[i].faceMat, CV_32F);
//         croppedFaces[i].faceMat = (croppedFaces[i].faceMat -
//         cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125; std::vector<cv::Mat>
//         temp; cv::split(croppedFaces[i].faceMat, temp); for (int i = 0; i <
//         temp.size(); i++) {
//             m_input.push_back(temp[i]);
//         }
//         croppedFaces[i].faceMat = m_input.clone();
//         m_input.release();
//     }
// }

std::vector<Object> ArcFace::postprocess(
    std::unordered_map<std::string, std::vector<float>> &feature_vectors,
    const DetectionParams &params,
    const std::vector<std::string> &detected_class) {
  std::vector<Object> num_faces;
  // int cnt = 0;
  // for (auto &chosenIdx : indices)
  // {
  //     if (cnt >= params.num_detect)
  //     {
  //         break;
  //     }
  //     Object face_box;
  //     face_box.probability = scores[chosenIdx];
  //     face_box.rect = bboxes[chosenIdx];
  //     face_box.label = 0;
  //     num_faces.push_back(face_box);
  //     cnt += 1;
  // }
  return num_faces;
}

}  // namespace tensorrt_inference