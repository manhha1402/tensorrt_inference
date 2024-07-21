#include "tensorrt_inference/yolov8.h"
#include <opencv2/cudaimgproc.hpp>


// Runs object detection on video stream then displays annotated results.
int main(int argc, char* argv[]) {
    tensorrt_inference::YoloV8Config config;
    std::string inputVideo = argv[2];
    const std::string home_dir = std::getenv("HOME");
    const std::string model_path  = home_dir + "/data/weights/";

	// Create the YoloV8 engine
	tensorrt_inference::YoloV8 yoloV8(model_path,argv[1], config);

	// Initialize the video stream
	cv::VideoCapture cap;

	// Open video capture
	try {
		cap.open(std::stoi(inputVideo));
	} catch (const std::exception& e) {
		cap.open(inputVideo);
	}

	// Try to use HD resolution (or closest resolution)
	auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl;

	if (!cap.isOpened())
		throw std::runtime_error("Unable to open video capture with input '" + inputVideo + "'");

	while (true) {
		// Grab frame
		cv::Mat img;
		cap >> img;

		if (img.empty())
			throw std::runtime_error("Unable to decode image from video stream.");

		// Run inference
		const auto objects = yoloV8.detectObjects(img);

		// Draw the bounding boxes on the image
		yoloV8.drawObjectLabels(img, objects);

		// Display the results
		cv::imshow("Object Detection", img);
		if (cv::waitKey(1) >= 0)
			break;
	}

	return 0;
}