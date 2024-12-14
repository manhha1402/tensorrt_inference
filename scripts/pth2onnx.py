import io
import numpy as np
import torch.onnx
import argparse
import cv2
img = cv2.imread("/home/manhha/github/cars.jpg")
img.resize()
resized_down = cv2.resize(img, (1344,1344), interpolation= cv2.INTER_LINEAR)
cv2.imwrite("/home/manhha/github/1344x1344.jpg",resized_down)
# parser = argparse.ArgumentParser(description='Process pt file.')
# parser.add_argument('--pt_path', help='path to pt file', required=True)
# args = parser.parse_args()


# model_state_dict = torch.load(args.pt_path)
# print(model_state_dict.keys())
# model_state_dict.eval()



