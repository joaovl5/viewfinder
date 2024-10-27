from ultralytics.engine.results import Results
from backend import DetectionResult, DetectionBox
from cache import Redis, TaskQueue
import datetime
import cv2


class ProcessedResult:
    data: str
    result: DetectionResult

    def __init__(self, result: DetectionResult, data: str) -> None:
        self.result = result
        self.data = data


class Hub:
    def __init__(self, uuid: str) -> None:
        self.redis = Redis()
        self.queue = TaskQueue()
        self.uuid = uuid
        self.enqueued = set()

    def plot_detection(self, detection: ProcessedResult) -> None:
        start_image = detection.result.orig_img
        for box in detection.result.boxes:
            x1, x2, x3, x4 = box.tuple()
            start_image = cv2.rectangle(start_image, (x1, x2), (x3, x4), (255, 0, 0), 4)
            start_image = cv2.putText(
                start_image,
                f"{self.search_box(box)} #{box.track_id} - {box.class_name} - {str(box.probability)}",
                (x1, x2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        flag, frame = cv2.imencode(".jpg", start_image)
        return frame

    @classmethod
    def make_box_key(cls, box: DetectionBox, uuid: str) -> str:
        return f"{uuid}:{box.track_id}"

    def search_box(self, box: DetectionBox):
        box_key = self.make_box_key(box, self.uuid)
        return self.redis.hget(box_key, "data")

    def process_detection(self, detection: DetectionResult) -> ProcessedResult:

        for box in detection.boxes:
            k = self.make_box_key(box, self.uuid)
            if k not in self.enqueued:
                self.enqueued.add(k)
                self.queue.enqueue_in(
                    time_delta=datetime.timedelta(seconds=10),
                    func=Hub.process_box,
                    args=(box, self.uuid),
                )

        return ProcessedResult(detection, "oiiii")

    @classmethod
    def process_box(cls, box: DetectionBox, uuid: str):
        box_key = cls.make_box_key(box, uuid)
        data = {"data": "eu sou uma " + box.class_name}
        Redis().hset(box_key, mapping=data)

    """
    Query cache for 
    """

    @classmethod
    def query_detection(cls, camera_id: int, track_id: int) -> None:
        pass
