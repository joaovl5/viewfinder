from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import numpy as np


class DetectionBox:
    x1: int
    x2: int
    x3: int
    x4: int
    class_name: str
    probability: float
    track_id: int
    data: str

    def __init__(self, x1: int, x2: int, x3: int, x4: int):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.data = ""

    def tuple(self):
        return (self.x1, self.x2, self.x3, self.x4)


class DetectionResult:
    orig_img: np.ndarray
    boxes: list[DetectionBox]


class DetectionBackend:
    def inference_frame(self, frame: np.ndarray) -> DetectionResult:
        pass


class YOLOBackend(DetectionBackend):
    def __init__(self, model_name="yolov8n.pt", conf=0.6) -> None:
        self.model = YOLO(model_name)
        self.conf_threshold = conf
        self.model.to("cuda")

    def inference_frame(self, frame: np.ndarray) -> DetectionResult:
        raw_result: Results = self.model.track(
            frame, verbose=False, conf=self.conf_threshold
        )[0]
        formatted_result = DetectionResult()
        formatted_result.orig_img = raw_result.orig_img
        formatted_result.boxes = list()
        for box in raw_result.boxes:
            x1, x2, x3, x4 = [int(value) for value in box.xyxy[0]]
            dbox = DetectionBox(x1, x2, x3, x4)
            dbox.class_name = self.model.names[int(box.cls)]
            dbox.probability = round(box.conf[0].item(), 2)
            dbox.track_id = int(box.id[0].item())
            formatted_result.boxes.append(dbox)
        return formatted_result


class RESNETBackend(DetectionBackend):
    def __init__(self) -> None:
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )
        self.model.to()

    def inference_frame(self, frame: np.ndarray) -> DetectionResult:
        inputs = self.processor(images=frame, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([frame.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        formatted_result = DetectionResult()
        formatted_result.orig_img = frame
        formatted_result.boxes = list()

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            dbox = DetectionBox(*[int(i) for i in box.tolist()])
            formatted_result.boxes.append(dbox)

        return formatted_result
