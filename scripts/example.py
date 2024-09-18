import cv2
import tensorrt_inference_py
from pathlib import Path
img_file = Path.home() / "data" / "test_images" / "image2.png"
image =cv2.imread(img_file.as_posix())
detector = tensorrt_inference_py.detection.Detection("facedetector","")
params = tensorrt_inference_py.detection.DetectionParams()
rect = tensorrt_inference_py.detection.Rect()


objects = detector.detect(image,params,['face'])
print(len(objects))
image1 = detector.draw(image,objects,params,2)
cv2.imwrite("result.png",image1)