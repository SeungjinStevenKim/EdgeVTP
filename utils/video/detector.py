"""Vehicle detection and tracking via Ultralytics YOLO + ByteTrack."""

import cv2
import numpy as np
from ultralytics import YOLO

from utils.inference_engine import resolve_device

_VEHICLE_CLASS_IDS = {2, 5, 7}


def _yolo_device(device_name):
    device = resolve_device(device_name)
    if device.type == "cuda":
        return str(device)
    if device.type == "mps":
        return "mps"
    return "cpu"


def _nms_boxes(boxes, scores, nms_threshold=0.5):
    if not boxes:
        return []
    rects = []
    for x1, y1, x2, y2 in boxes:
        rects.append([int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))])
    indices = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.0, nms_threshold=nms_threshold)
    if len(indices) == 0:
        return []
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    return [boxes[int(i)] for i in indices]


class YoloDetector:
    def __init__(
        self,
        model_path="yolov8n.pt",
        conf_threshold=0.25,
        nms_threshold=0.45,
        max_detections=50,
        imgsz=1280,
        device="auto",
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.imgsz = imgsz
        self.device = _yolo_device(device)

    def detect(self, frame):
        """Return list of (x1, y1, x2, y2) boxes."""
        tracked = self.track(frame)
        return [bbox for _, bbox in tracked]

    def track(self, frame):
        """Return list of (track_id, (x1, y1, x2, y2)) using YOLO ByteTrack."""
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            max_det=self.max_detections,
            imgsz=self.imgsz,
            device=self.device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        outputs = []
        for box in results[0].boxes:
            class_id = int(box.cls[0].cpu().numpy())
            if class_id not in _VEHICLE_CLASS_IDS:
                continue
            if box.id is None:
                continue
            track_id = int(box.id[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            outputs.append((track_id, (float(x1), float(y1), float(x2), float(y2))))
        return outputs


def build_detector(args):
    return YoloDetector(
        model_path=getattr(args, "model_path", "yolov8n.pt"),
        conf_threshold=getattr(args, "conf_threshold", 0.30),
        nms_threshold=getattr(args, "nms_threshold", 0.45),
        max_detections=getattr(args, "max_detections", 50),
        imgsz=getattr(args, "yolo_imgsz", 1280),
        device=getattr(args, "device", "auto"),
    )
