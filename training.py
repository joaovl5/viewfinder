from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model.to("cuda")
results = model.train(data="coco8.yaml", epochs=1, imgsz=640)
