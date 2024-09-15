#include <numeric>

#include "tensorrt_inference/tensorrt_inference.h"

void preprocessFace(cv::Mat &face, cv::Mat &output) {
  cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
  face.convertTo(face, CV_32F);
  face = (face - cv::Scalar(127.5, 127.5, 127.5)) * 0.0078125;
  std::vector<cv::Mat> temp;
  cv::split(face, temp);
  for (int i = 0; i < temp.size(); i++) {
    output.push_back(temp[i]);
  }
}

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
  // tensorrt_inference::ModelParams params(0.2,0.2,20);
  //  Run inference
  tensorrt_inference::DetectionParams params(0.5, 0.2, 0.5, 0.5, 10);
  auto faces1 = retinaface.detect(img1, params);
  cv::Mat result = img1.clone();
  retinaface.drawObjectLabels(result, faces1);
  // Save the image to disk
  auto outputName =
      inputImage1.substr(0, inputImage1.find_last_of('.')) + "_annotated.jpg";
  cv::imwrite(outputName, result);
  std::cout << "Saved annotated image to: " << outputName << std::endl;

  auto faces2 = retinaface.detect(img2, params);
  result = img2.clone();
  retinaface.drawObjectLabels(result, faces2);
  // Save the image to disk
  outputName =
      inputImage2.substr(0, inputImage2.find_last_of('.')) + "_annotated.jpg";
  cv::imwrite(outputName, result);
  std::cout << "Saved annotated image to: " << outputName << std::endl;

  tensorrt_inference::ArcFace arcface("arcfaceresnet100-8");
  auto cropped_faces1 = arcface.getCroppedFaces(img1, faces1);
  auto cropped_faces2 = arcface.getCroppedFaces(img2, faces2);

  cv::imwrite("crop_face1.png", cropped_faces1[0].face_mat);
  cv::imwrite("crop_face2.png", cropped_faces2[0].face_mat);

  std::unordered_map<std::string, std::vector<float>> feature_vectors;
  // cv::Mat input1,input2;
  // arcface.preprocessFace(cropped_faces1[0].face_mat,input1);
  // arcface.preprocessFace(cropped_faces2[0].face_mat,input2);
  // cv::cuda::GpuMat gpu_input1, gpu_input2;
  // gpu_input1.upload(input1);
  // gpu_input2.upload(input2);

  // bool ret = arcface.m_trtEngine->runInference(gpu_input1,feature_vectors);
  // cv::Mat out1 = cv::Mat(512, 1, CV_32F, feature_vectors.at("fc1").data());
  // cv::Mat out_norm1;
  // cv::normalize(out1, out_norm1);
  // std::cout<< feature_vectors.at("fc1")[0]<<std::endl;

  // feature_vectors.clear();
  // ret = arcface.m_trtEngine->runInference(gpu_input2,feature_vectors);
  // std::cout<< feature_vectors.at("fc1")[0]<<std::endl;
  // cv::Mat out2 = cv::Mat(512, 1, CV_32F, feature_vectors.at("fc1").data());
  // cv::Mat out_norm2;
  // cv::normalize(out2, out_norm2);

  cv::cuda::GpuMat gpu_input1(cropped_faces1[0].face_mat);
  bool ret = arcface.doInference(gpu_input1, feature_vectors);
  cv::Mat out1 = cv::Mat(512, 1, CV_32F, feature_vectors.at("fc1").data());
  cv::Mat out_norm1;
  cv::normalize(out1, out_norm1);

  cv::cuda::GpuMat gpu_input2(cropped_faces2[0].face_mat);
  ret = arcface.doInference(gpu_input2, feature_vectors);

  cv::Mat out2 = cv::Mat(512, 1, CV_32F, feature_vectors.at("fc1").data());
  cv::Mat out_norm2;
  cv::normalize(out2, out_norm2);
  cv::Mat res = out_norm1.t() * out_norm2;
  std::cout << "similarity score: " << *(float *)res.data << std::endl;

  return 0;
}