# import cv2
# import tensorrt_inference_py
# from pathlib import Path
# img_file = Path.home() / "data" / "test_images" / "image2.png"
# image =cv2.imread(img_file.as_posix())
# detector = tensorrt_inference_py.detection.Detection("facedetector","")
# params = tensorrt_inference_py.detection.DetectionParams()
# rect = tensorrt_inference_py.detection.Rect()


# objects = detector.detect(image,params,['face'])
# print(len(objects))
# image1 = detector.draw(image,objects,params,2)
# cv2.imwrite("result.png",image1)
# # import yaml



# import yaml
# def convert_txt_to_yaml(txt_file, yaml_file):

#     with open(txt_file, 'r', encoding='utf-8') as file:
#         lines = file.readlines()

#     # Create a dictionary with index numbers as keys
#     data = {i+1: line.strip() for i, line in enumerate(lines) if line.strip()}

#     # Write the dictionary to a YAML file without quotes
#     with open(yaml_file, 'w', encoding='utf-8') as yaml_out:
#         yaml.dump(data, yaml_out, allow_unicode=True, default_style=None)

# # Example usage
# convert_txt_to_yaml('/home/neura_ai/data/weights/paddleocr/ppocr_keys_v1.txt', '/home/neura_ai/data/weights/paddleocr/output.yaml')

# print("Conversion complete! Check the 'output.yaml' file.")