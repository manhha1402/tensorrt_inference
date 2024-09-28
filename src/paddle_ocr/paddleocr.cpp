#include "tensorrt_inference/paddle_ocr/paddleocr.h"

namespace {

static void get_resize_ratio(int w, int h, int max_size_len, int& resize_h,
                             int& resize_w) {
  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = float(max_size_len) / float(h);
    } else {
      ratio = float(max_size_len) / float(w);
    }
  }

  resize_h = int(float(h) * ratio);
  resize_w = int(float(w) * ratio);

  resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
  resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);
}
void resizeImgType0(const cv::cuda::GpuMat& img, cv::cuda::GpuMat& resize_img,
                    int max_size_len, float& ratio_h, float& ratio_w) {
  int resize_w;
  int resize_h;

  get_resize_ratio(img.cols, img.rows, max_size_len, resize_h, resize_w);

  cv::cuda::resize(img, resize_img, cv::Size(resize_w, resize_h));
  ratio_h = float(resize_h) / float(img.rows);
  ratio_w = float(resize_w) / float(img.cols);
}
static std::vector<int> argsort(const std::vector<float>& array) {
  const int array_len(array.size());
  std::vector<int> array_index(array_len, 0);
  for (int i = 0; i < array_len; ++i) array_index[i] = i;

  std::sort(
      array_index.begin(), array_index.end(),
      [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

  return array_index;
}

static std::vector<std::string> readDict(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  return m_vec;
}

}  // namespace
namespace tensorrt_inference {

TextDetection::TextDetection(const std::string& model_name,
                             tensorrt_inference::Options options,
                             const std::filesystem::path& model_dir)
    : Engine(options) {
  std::string config_file =
      (model_dir / model_name / "text_detection_config.yaml").string();
  YAML::Node config = YAML::LoadFile(config_file);
  onnx_file_ =
      (model_dir / model_name / config["onnx_file"].as<std::string>()).string();
  auto succ = buildLoadNetwork(onnx_file_);
  if (!succ) {
    const std::string errMsg =
        "Error: Unable to build or load the TensorRT engine. "
        "Try increasing TensorRT log severity to kVERBOSE (in "
        "/libs/tensorrt-cpp-api/engine.cpp).";
    throw std::runtime_error(errMsg);
  }
}
uint32_t TextDetection::getMaxOutputLength(
    const nvinfer1::Dims& tensorShape) const {
  return m_options.MAX_DIMS_[2] * m_options.MAX_DIMS_[3];
}
void TextDetection::runInference(
    const cv::cuda::GpuMat& gpu_img,
    std::vector<std::vector<std::vector<int>>>& boxes) {
  boxes.clear();

  ////////////////////// preprocess ////////////////////////
  float ratio_h{};  // = resize_h / h
  float ratio_w{};  // = resize_w / w
  cv::cuda::GpuMat resize_img;
  resizeImgType0(gpu_img, resize_img, max_side_len_, ratio_h, ratio_w);
  normalizeOp(resize_img, mean_, scale_);
  permute(resize_img);
  size_t batchSize = m_options.optBatchSize;
  cudaStream_t inferenceCudaStream;
  Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

  nvinfer1::Dims4 inputDims2 = {(int64_t)batchSize, 3, resize_img.rows,
                                resize_img.cols};
  m_context->setInputShape(input_map_.begin()->first.c_str(), inputDims2);

  if (!m_context->allInputDimensionsSpecified()) {
    auto msg = "Error, not all required dimensions specified.";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }

  bool status = m_context->setTensorAddress(input_map_.begin()->first.c_str(),
                                            resize_img.ptr<void>());
  status = m_context->setTensorAddress(output_map_.begin()->first.c_str(),
                                       output_map_.begin()->second.buffer);

  m_context->enqueueV3(inferenceCudaStream);

  cv::cuda::GpuMat result(resize_img.rows, resize_img.cols, CV_32FC1,
                          output_map_.begin()->second.buffer);

  cv::Mat pred;
  result.download(pred);

  cv::Mat bitmap;

  cv::threshold(pred, bitmap, det_db_thresh_, 1, cv::THRESH_BINARY);

  bitmap.convertTo(bitmap, CV_8UC1, 255.0f);
  Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
  Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

  boxes = BoxesFromBitmap(pred, bitmap, this->det_db_box_thresh_,
                          this->det_db_unclip_ratio_, this->use_polygon_score_);
  boxes = FilterTagDetRes(boxes, ratio_h, ratio_w, gpu_img);
}

/////////////////////////////////////////////////////////////////////////////////
TextRecognition::TextRecognition(const std::string& model_name,
                                 tensorrt_inference::Options options,
                                 const std::filesystem::path& model_dir)
    : Engine(options) {
  std::string config_file =
      (model_dir / model_name / "text_recognition_config.yaml").string();
  YAML::Node config = YAML::LoadFile(config_file);

  onnx_file_ =
      (model_dir / model_name / config["onnx_file"].as<std::string>()).string();
  std::cout << onnx_file_ << std::endl;

  if (config["normalized"]) {
    normalized_ = config["normalized"].as<bool>();
  }
  if (config["swapBR"]) {
    swapBR_ = config["swapBR"].as<bool>();
  }
  if (config["sub_vals"]) {
    sub_vals_ = config["sub_vals"].as<std::vector<float>>();
  }
  if (config["div_vals"]) {
    div_vals_ = config["div_vals"].as<std::vector<float>>();
  }

  if (config["labels_file"]) {
    std::string labels_file =
        (model_dir / model_name / config["labels_file"].as<std::string>())
            .string();
    if (Util::doesFileExist(std::filesystem::path(labels_file))) {
      label_list_ = readDict(labels_file);
      label_list_.insert(label_list_.begin(), "#");
      label_list_.push_back(" ");
    } else {
      spdlog::error("label file is not existed!");
    }
  }
  auto succ = buildLoadNetwork(onnx_file_);
  if (!succ) {
    const std::string errMsg =
        "Error: Unable to build or load the TensorRT engine. "
        "Try increasing TensorRT log severity to kVERBOSE (in "
        "/libs/tensorrt-cpp-api/engine.cpp).";
    throw std::runtime_error(errMsg);
  }
}
uint32_t TextRecognition::getMaxOutputLength(
    const nvinfer1::Dims& tensorShape) const {
  return (m_options.MAX_DIMS_[3] + 4) / 8 *
         tensorShape.d[tensorShape.nbDims - 1] * rec_batch_num_;
}

std::pair<std::string, double> TextRecognition::runInference(
    std::vector<cv::cuda::GpuMat>& img_list) {
  int img_num = img_list.size();
  std::vector<std::pair<std::vector<std::string>, double>> rec_res;
  std::vector<int> idx_map;
  std::vector<float>
      width_list;  // Store the aspect ratio of all images to be recognized
  for (int i = 0; i < img_num; i++) {
    width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
  }
  // Sort the aspect ratios from small to
  // large and get the indices
  std::vector<int> indices = argsort(width_list);
  std::vector<int> copy_indices = indices;
  // Record the idx of the empty recognition result in a batch
  std::vector<int> nan_idx;
  for (int begin_img = 0; begin_img < img_num; begin_img += rec_batch_num_) {
    /////////////////////////// preprocess ///////////////////////////////
    auto preprocess_start = std::chrono::steady_clock::now();
    int end_img = std::min(img_num, begin_img + rec_batch_num_);

    float max_wh_ratio = 0;
    for (int ino = begin_img; ino < end_img; ino++) {
      int h = img_list[indices[ino]].rows;
      int w = img_list[indices[ino]].cols;
      float wh_ratio = w * 1.0 / h;
      max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
    }  // Find the largest aspect ratio
    std::vector<cv::cuda::GpuMat> norm_img_batch;
    for (int ino = begin_img; ino < end_img; ino++) {
      cv::cuda::GpuMat resize_img;
      resizeOp(img_list[indices[ino]], resize_img, max_wh_ratio);
      normalizeOp(resize_img, mean_, scale_);
      norm_img_batch.push_back(resize_img);
    }  // Resize the imgs in a batch to a height of 48 and a width of
       // 48*max_wh_ratio according to the maximum aspect ratio, and normalize
       // them respectively.
    //////////////////////////
    // The width of the images in this batch
    int batch_width = int(m_options.OPT_DIMS_[2] * max_wh_ratio);

    cv::cuda::GpuMat dest;
    permuteBatchOp(norm_img_batch, dest);
    nvinfer1::Dims4 inputDims2 = {(int64_t)norm_img_batch.size(), 3,
                                  m_options.OPT_DIMS_[2], batch_width};
    const std::string& input_tensor_name = input_map_.begin()->first;
    const std::string& output_tensor_name = output_map_.begin()->first;

    m_context->setInputShape(input_tensor_name.c_str(), inputDims2);
    // Create stream
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
    // m_context->nb
    auto tensorShape = output_map_.begin()->second.dims;

    size_t outputWidth = (batch_width + 4) / 8;
    size_t outputLength = outputWidth * tensorShape.d[2];
    bool status = m_context->setTensorAddress(input_tensor_name.c_str(),
                                              dest.ptr<void>());
    status = m_context->setTensorAddress(output_tensor_name.c_str(),
                                         output_map_.begin()->second.buffer);

    m_context->enqueueV3(inferenceCudaStream);

    std::vector<std::vector<float>> result;
    for (int img = 0; img < norm_img_batch.size(); ++img) {
      std::vector<float>& output = result.emplace_back(outputLength);
      output.resize(outputLength);

      Util::checkCudaErrorCode(cudaMemcpyAsync(
          output.data(),
          static_cast<char*>(output_map_.begin()->second.buffer) +
              (img * sizeof(float) * outputLength),
          outputLength * sizeof(float), cudaMemcpyDeviceToHost,
          inferenceCudaStream));
    }

    cudaStreamSynchronize(inferenceCudaStream);
    cudaStreamDestroy(inferenceCudaStream);

    std::vector<int> predict_shape;
    predict_shape.push_back(norm_img_batch.size());
    predict_shape.push_back(outputWidth);
    predict_shape.push_back(tensorShape.d[2]);

    for (int m = 0; m < predict_shape[0]; m++) {  // m = batch_size
      std::pair<std::vector<std::string>, double> temp_box_res;
      std::vector<std::string> str_res;
      int argmax_idx;
      int last_index = 0;
      float score = 0.f;
      int count = 0;
      float max_value = 0.0f;

      std::vector<float>& imgRes = result[m];
      for (int n = 0; n < predict_shape[1]; n++) {  // n = 2*l + 1
        argmax_idx = int(argmax(imgRes.cbegin() + n * predict_shape[2],
                                imgRes.cbegin() + (n + 1) * predict_shape[2]));
        max_value = float(
            *std::max_element(imgRes.cbegin() + n * predict_shape[2],
                              imgRes.cbegin() + (n + 1) * predict_shape[2]));

        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
          score += max_value;
          count += 1;
          str_res.push_back(label_list_[argmax_idx]);
        }
        last_index = argmax_idx;
      }
      score /= count;
      if (isnan(score)) {
        nan_idx.push_back(begin_img + m);
        continue;
      }

      temp_box_res.first = str_res;
      temp_box_res.second = score;
      rec_res.push_back(temp_box_res);
    }
  }

  for (int i = nan_idx.size() - 1; i >= 0; i--) {
    copy_indices.erase(copy_indices.begin() + nan_idx[i]);
  }

  if (copy_indices.size() == rec_res.size()) {
    idx_map = copy_indices;
  }
  std::string result;
  double score = 0.0;
  for (int n = 0; n < rec_res.size(); n++) {
    std::string res;
    std::vector<std::string> temp_res = rec_res[n].first;

    for (int m = 0; m < temp_res.size(); m++) {
      res += temp_res[m];
    }
    result += " " + res;
    score += rec_res[n].second;
  }
  score /= rec_res.size();
  return std::pair(result, score);
}

PaddleOCR::PaddleOCR(const std::string& model_name,
                     tensorrt_inference::Options options_det,
                     tensorrt_inference::Options options_rec,
                     const std::filesystem::path& model_dir) {
  text_det_ =
      std::make_shared<TextDetection>(model_name, options_det, model_dir);
  text_rec_ =
      std::make_shared<TextRecognition>(model_name, options_rec, model_dir);
}

std::pair<std::string, double> PaddleOCR::runInference(
    cv::Mat& img, const CroppedObject& cropped_plate) {
  const cv::cuda::GpuMat gpu_img(cropped_plate.croped_object);
  std::vector<std::vector<std::vector<int>>> boxes;
  text_det_->runInference(gpu_img, boxes);
  if (!boxes.empty()) {
    std::vector<cv::cuda::GpuMat> img_list;
    for (int j = 0; j < boxes.size(); j++) {
      cv::cuda::GpuMat crop_img;
      crop_img = getRotateCropImage(gpu_img, boxes[j]);
      cv::Mat tmp;
      crop_img.download(tmp);
      img_list.push_back(crop_img);
    }
    return text_rec_->runInference(img_list);
  } else {
    return std::pair("", 0.0);
  }
}
cv::Mat PaddleOCR::drawBBoxLabels(const cv::Mat& image,
                                  const std::vector<CroppedObject>& objects,
                                  unsigned int scale) {
  cv::Mat result = image.clone();
  for (const auto& object : objects) {
    cv::Scalar color = cv::Scalar(0, 255, 0);
    float meanColor = cv::mean(color)[0];
    cv::Scalar txtColor;
    if (meanColor > 0.5) {
      txtColor = cv::Scalar(0, 0, 0);
    } else {
      txtColor = cv::Scalar(255, 255, 255);
    }
    const auto& rect = object.rect;
    // Draw rectangles and text
    char text[256];
    sprintf(text, "%s %.1f%%", object.label.c_str(), object.probability * 100);

    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                         0.35 * scale, scale, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7 * 255;

    int x = object.rect.x;
    int y = object.rect.y + 1;

    cv::rectangle(result, rect, color * 255, scale + 1);

    cv::rectangle(
        result,
        cv::Rect(cv::Point(x, y),
                 cv::Size(labelSize.width, labelSize.height + baseLine)),
        txt_bk_color, -1);

    cv::putText(result, text, cv::Point(x, y + labelSize.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);
  }
  return result;
}
}  // namespace tensorrt_inference