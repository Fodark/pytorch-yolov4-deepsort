from .YOLOV4 import yolo
from .YOLOV4.yolo import YOLO
from .PyTorch_YOLOv4.utils.inference_model import YoloMish


__all__ = ['build_detector']

def build_detector(finetuned=False, use_cuda=True):
    if finetuned:
        return YoloMish()
    else:
        return YOLO(use_cuda)
