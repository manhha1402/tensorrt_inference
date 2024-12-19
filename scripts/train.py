from ultralytics import YOLO
import cv2
img = cv2.imread("/home/neura_ai/data/cars.jpg")
resized = cv2.resize(img,(1344,1344))
cv2.imwrite("/home/neura_ai/data/resize.jpg",resized)
# # Load a model
#model = YOLO("yolov9e-seg.pt")  # load a pretrained model (recommended for training)
#print(model.names) 
#model.save
#path = model.export(format="onnx")
# # # Use the model
# # model.train(data="/home/manhhahoang/data/fabric_defect_detection.v1i.yolov8/data.yaml", epochs=300)  # train the model
# # metrics = model.val()  # evaluate model performance on the validation set
# # results = model("/home/manhhahoang/data/fabric_defect_detection.v1i.yolov8/test/images/z5549753700613_e29abad32aa9304c32fdea6f45750c78_jpg.rf.72abc507ab6951327aaf72e85f9468c3.jpg")  # predict on an image
# # path = model.export(format="onnx")  # export the model to ONNX format