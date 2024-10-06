import cv2
import tensorrt_inference_py
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

# from deepface import DeepFace

img_file1 = Path.home() / "data" / "test_images" / "david1.jpg"
img_file2 = Path.home() / "data" / "test_images" / "david2.jpg"

image1 =cv2.imread(img_file1.as_posix())
image2 =cv2.imread(img_file2.as_posix())

detector = tensorrt_inference_py.detection.Detection("facedetector")
params = tensorrt_inference_py.detection.DetectionParams()
# rect = tensorrt_inference_py.detection.Rect()
objects1 = detector.detect(image1,params,['face'])
objects2 = detector.detect(image2,params,['face'])

# image1 = detector.draw(image,objects,params,1)
# cv2.imwrite("result.png",image1)
# print(objects[0].rect)
cropped_image1 = image1[objects1[0].rect.y:objects1[0].rect.y+objects1[0].rect.height, objects1[0].rect.x:objects1[0].rect.x+objects1[0].rect.width]
cropped_image2 = image2[objects2[0].rect.y:objects2[0].rect.y+objects2[0].rect.height, objects2[0].rect.x:objects2[0].rect.x+objects2[0].rect.width]

cv2.imwrite("cropped_image1.png",cropped_image1)
cv2.imwrite("cropped_image2.png",cropped_image2)
facenet = tensorrt_inference_py.model.Model("FaceNet_vggface2_optmized")
embedding1 = facenet.get_embedding(cropped_image1)
embedding2 = facenet.get_embedding(cropped_image2)

C = np.matmul(embedding1.transpose(), embedding2)  # Or A @ B
D = embedding1.transpose() @ embedding2  # Or A @ B

print(C)
print(D)











# import cv2

# stream_url = "rtsp://admin:namtiep2005@192.168.1.125:554/Streaming/Channels/101"
# cap = cv2.VideoCapture(stream_url)

# while True:
#     ret, frame = cap.read()
#     print(frame.shape)
#     if ret:
#         cv2.imshow('Hikvision Stream', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()


