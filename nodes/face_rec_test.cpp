#include <numeric>

#include "tensorrt_inference/tensorrt_inference.h"

// Runs object detection on an input image then saves the annotated image to
// disk.
int main(int argc, char *argv[]) {
  std::string inputImage1 = argv[1];
  std::string inputImage2 = argv[2];
  tensorrt_inference::RetinaFace retinaface("facedetector");
  // Read the input image
  auto img1 = cv::imread(inputImage1);
  if (img1.empty()) {
    std::cout << "Error: Unable to read image at path '" << inputImage1 << "'"
              << std::endl;
    return -1;
  }
  auto img2 = cv::imread(inputImage2);
  if (img2.empty()) {
    std::cout << "Error: Unable to read image at path '" << inputImage2 << "'"
              << std::endl;
    return -1;
  }

  tensorrt_inference::DetectionParams params(0.5, 0.2, 0.5, 0.5, 10);
  auto faces1 = retinaface.detect(img1, params);
  cv::Mat result = img1.clone();
  retinaface.drawObjectLabels(result, faces1);
  auto faces2 = retinaface.detect(img2, params);
  result = img2.clone();
  retinaface.drawObjectLabels(result, faces2);

  tensorrt_inference::FaceRecognition face_rec("FaceNet_vggface2_optmized");
  const int h = face_rec.m_trtEngine->getInputInfo().begin()->second.dims.d[2];
  const int w = face_rec.m_trtEngine->getInputInfo().begin()->second.dims.d[3];

  auto cropped_faces1 =
      tensorrt_inference::getCroppedObjects(img1, faces1, w, h, false);
  auto cropped_faces2 =
      tensorrt_inference::getCroppedObjects(img2, faces2, w, h, false);

  cv::imwrite("crop_face1.png", cropped_faces1[0].croped_object);
  cv::imwrite("crop_face2.png", cropped_faces2[0].croped_object);
  std::unordered_map<std::string, std::vector<float>> feature_vectors1,
      feature_vectors2;
  cv::cuda::GpuMat gpu_input1(cropped_faces1[0].croped_object);
  bool ret = face_rec.doInference(gpu_input1, feature_vectors1);
  cv::Mat out1 = cv::Mat(feature_vectors1.begin()->second.size(), 1, CV_32F,
                         feature_vectors1.begin()->second.data());
  cv::Mat out_norm1;
  cv::normalize(out1, out_norm1);

  cv::cuda::GpuMat gpu_input2(cropped_faces2[0].croped_object);
  ret = face_rec.doInference(gpu_input2, feature_vectors2);

  cv::Mat out2 = cv::Mat(feature_vectors1.begin()->second.size(), 1, CV_32F,
                         feature_vectors2.begin()->second.data());
  cv::Mat out_norm2;
  cv::normalize(out2, out_norm2);

  std::cout << "similarity" << std::endl;
  cv::Mat res = out_norm1.t() * out_norm2;

  std::cout << "similarity score: " << *(float *)res.data << std::endl;
  return 0;
}