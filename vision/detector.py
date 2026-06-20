from ultralytics import YOLO
import cv2
import torch
from dataclasses import dataclass
from typing import List

@dataclass
class Detection:
    track_id: int
    bbox: list        # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    frame_idx: int

class VehicleDetector:
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self, weights="yolov8n.pt", device=None, conf=0.4):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)
        self.conf = conf

    def detect_frame(self, frame, frame_idx: int) -> List[Detection]:
        results = self.model(frame, conf=self.conf,
                             device=self.device, verbose=False)
        detections = []
        for box in results[0].boxes:
            cls = self.model.names[int(box.cls)]
            if cls not in self.VEHICLE_CLASSES:
                continue
            detections.append(Detection(
                track_id=-1,
                bbox=box.xyxy[0].tolist(),
                confidence=float(box.conf),
                class_name=cls,
                frame_idx=frame_idx,
            ))
        return detections

    def detect_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        all_detections = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            dets = self.detect_frame(frame, frame_idx)
            all_detections.extend(dets)
            frame_idx += 1
        cap.release()
        return all_detections