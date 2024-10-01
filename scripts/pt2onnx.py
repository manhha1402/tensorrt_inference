# from ultralytics import YOLO
# import argparse
# import yaml
# import pathlib
# parser = argparse.ArgumentParser(description='Process pt file.')
# parser.add_argument('--pt_path', help='path to pt file', required=True)
# args = parser.parse_args()

# # TODO: Specify which model you want to convert
# # Model can be downloaded from https://github.com/ultralytics/ultralytics
# model = YOLO(args.pt_path)
# model_dir = pathlib.Path(args.pt_path).parent
# class_list_file =  pathlib.Path(args.pt_path).stem
# with open( (model_dir / class_list_file).as_posix() + ".yaml" , 'w') as f:
#     yaml.dump(model.names, f, default_flow_style=False)
# model.fuse()
# model.info(verbose=False)  # Print model information
# model.export(format="onnx", simplify=True)


import onnx
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1

torch_model = InceptionResnetV1('vggface2').eval()
torch_input = torch.randn(1, 3, 160, 160)
torch.onnx.export(torch_model, torch_input,
                  'facenet.onnx',
                  opset_version=18,  # The ONNX version to export the model to
                  do_constant_folding=True,  # Whether to execute constant folding for optimization
                  input_names=['input0'],  # The model's input names
                  output_names=['output0']  # The model's output names
                  )

# Load the ONNX model
onnx_model = onnx.load('facenet.onnx')

# Check that the model is well-formed
onnx.checker.check_model(onnx_model)

print("ONNX model is well-formed and saved successfully.")


