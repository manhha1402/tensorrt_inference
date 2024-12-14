from ultralytics import YOLO
import argparse
import yaml
import pathlib
parser = argparse.ArgumentParser(description='Process pt file.')
parser.add_argument('--pt_path', help='path to pt file', required=True)
args = parser.parse_args()

# TODO: Specify which model you want to convert
# Model can be downloaded from https://github.com/ultralytics/ultralytics
model = YOLO(args.pt_path)
model_dir = pathlib.Path(args.pt_path).parent
class_list_file =  pathlib.Path(args.pt_path).stem
with open( (model_dir / class_list_file).as_posix() + ".yaml" , 'w') as f:
    yaml.dump(model.names, f, default_flow_style=False)
model.fuse()
model.info(verbose=False)  # Print model information
model.export(format="onnx", simplify=True)


