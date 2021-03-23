from .YOLOV4 import yolo
from .YOLOV4.yolo import YOLO
from .PyTorch_YOLOv4.utils.inference_model import YoloMish


__all__ = ['build_detector']

def build_detector(finetuned=False, use_cuda=True, cfg="models/yolov4l-mish.yaml", weights="weights/yolov4l-mish.pt"):
    if finetuned:
        return YoloMish(cfg=cfg, weights=weights)
    else:
        return YOLO(use_cuda)
