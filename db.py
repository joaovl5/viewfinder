from mongoengine import *


class Camera(Document):
    source = ListField(required=True)
    name = StringField(required=True, max_length=50)


class Database:
    def __init__(self) -> None:
        connect("viewfinder")

    def clear_cameras(self):
        for cam in Camera.objects:
            cam.delete()

    def setup_test_cameras(self):
        self.clear_cameras()
        test_cameras = [
            {"source": [0], "name": "webcam"},
            # {"source": ["http://192.168.100.2:4747/video"], "name": "webcam"},
        ]
        [Camera(**cam).save() for cam in test_cameras]

    def get_cameras(self):
        return Camera.objects
