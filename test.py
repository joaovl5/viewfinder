from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
from math import dist
from enum import Enum
import time
from threading import Thread

from ultralytics import YOLO


class ConfigMapping(Enum):
    MOVE_FORWARD = "w"
    MOVE_LEFT = "a"
    MOVE_BACKWARD = "s"
    MOVE_RIGHT = "d"
    MOVE_JUMP = Key.space
    MOVE_SILENT = Key.shift_l
    MOVE_CROUCH = Key.ctrl_l
    MOUSE_FIRE = Button.left
    MOUSE_ADS = Button.right
    MOUSE_SWITCH = Button.scroll_down


class Movement(Enum):
    FORWARD = "MOVE_FORWARD"
    LEFT = "MOVE_LEFT"
    BACKWARD = "MOVE_BACKWARD"
    RIGHT = "MOVE_RIGHT"
    JUMP = "MOVE_JUMP"
    SILENT = "MOVE_SILENT"
    CROUCH = "MOVE_CROUCH"


class InputService:
    def __init__(self) -> None:
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

    def begin_movement(self, movement: Movement) -> None:
        self.keyboard.press(ConfigMapping[movement.value])

    def stop_movement(self, movement: Movement) -> None:
        self.keyboard.release(ConfigMapping[movement.value])

    def mouse_move(self, dx: int, dy: int) -> None:
        self.mouse.move(dx, dy)

    def mouse_fire(self) -> None:
        self.mouse.click(ConfigMapping.MOUSE_FIRE)

    def mouse_fire_press(self) -> None:
        self.mouse.press(ConfigMapping.MOUSE_FIRE)

    def mouse_fire_release(self) -> None:
        self.mouse.release(ConfigMapping.MOUSE_FIRE)

    def mouse_ads(self) -> None:
        self.mouse.click(ConfigMapping.MOUSE_ADS)

    def mouse_switch(self) -> None:
        self.mouse.click(ConfigMapping.MOUSE_SWITCH)


lerp = 0.05


def calculate_pos(xyxy):
    left, top, right, bottom = xyxy
    width = 480
    height = 640
    return int((((left + right) / 2) + 1) * lerp), int((top + bottom) / 2 * lerp)


def thread_safe_predict(process_signal):
    local_model = YOLO("yolov8n.pt")
    local_model.to("cuda")
    results = local_model.predict(0, stream=True, conf=0.5, iou=0.5, half=True)

    for r in results:
        for box in r.boxes:

            label = local_model.names[int(box.cls)]
            if label == "person":
                b = box.xyxy[0]
                x, y = calculate_pos(b)
                print(x, y)
                process_person(x, y)


svc = InputService()


def process_person(x, y):
    svc.mouse_move(x, y)


thr = Thread(target=thread_safe_predict, args=(process_person,))
thr.start()
