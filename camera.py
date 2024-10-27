from ultralytics import YOLO
from cache import Queue
from hub import Hub
from backend import *
import cv2
import time
import traceback
import threading
import uuid


class Camera:
    queue: Queue

    def __init__(self, source) -> None:
        self.source = source
        self.backend = YOLOBackend()
        self.queue = Queue()
        self.active = False
        self.uuid = uuid.uuid4().hex
        self.hub = Hub(self.uuid)

    def thread_work(self, queue: Queue) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Cannot open source")

        while True:
            success, frame = cap.read()
            try:
                if success:
                    results = self.backend.inference_frame(frame)

                    if self.active:
                        processed_results = self.hub.process_detection(results)
                        plotted_frame = self.hub.plot_detection(processed_results)
                        queue.put(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + bytearray(plotted_frame)
                            + b"\r\n"
                        )

            except Exception as ex:
                print("bugug")
                traceback.format_exc(ex)

        # results = self.model.track(
        #     source=self.source, stream=True, show=True, stream_buffer=False, device=0
        # )  # predict on an image

        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     probs = result.probs  # Probs object for classification outputs
        #     obb = result.obb  # Oriented boxes object for OBB outputs

    def start(self) -> None:
        self.process = threading.Thread(target=self.thread_work, args=(self.queue,))
        self.process.start()

    def join(self) -> None:
        self.process.join()

    def stop(self) -> None:
        self.process.kill()

    def activate(self) -> None:
        self.active = True
        self.last_activation_time = int(time.time())

    def deactivate(self) -> None:
        self.active = False


class Pool:
    def __init__(self, sources=[]) -> None:
        self.cameras = [Camera(source) for source in sources]
        self.background_tasks: dict[str, threading.Timer] = dict()

    def activation_check(self, camera_id: int) -> None:
        diff = int(time.time()) - self.cameras[camera_id].last_activation_time
        if diff > 5:
            self.deactivate_camera(camera_id)

    def add(self, *sources) -> None:
        self.cameras.append([Camera(source) for source in sources])

    def activate_camera(self, camera_id):
        if camera_id in self.background_tasks:
            self.background_tasks[camera_id].cancel()
        self.cameras[camera_id].activate()
        task = threading.Timer(
            interval=10, function=self.activation_check, args=(camera_id,)
        )
        task.start()
        self.background_tasks[camera_id] = task

    def deactivate_camera(self, camera_id):
        self.cameras[camera_id].deactivate()

    def get_frame(self, camera_id: int):
        while True:
            self.activate_camera(camera_id)
            val = self.cameras[camera_id].queue.get()
            yield val

    def start(self) -> None:
        [c.start() for c in self.cameras]

    def join(self) -> None:
        [c.join() for c in self.cameras]

    def stop(self) -> None:
        [c.stop() for c in self.cameras]
