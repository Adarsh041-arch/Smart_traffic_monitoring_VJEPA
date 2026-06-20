from ultralytics import YOLO
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict

@dataclass
class TrackedVehicle:
    track_id: int
    class_name: str
    bbox: list
    frame_idx: int
    confidence: float

class VehicleTracker:
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self, weights="yolov8n.pt", device=None):
        self.model = YOLO(weights)
        self.device = device

    def track_video(self, video_path: str) -> Dict[int, List[TrackedVehicle]]:
        """
        Returns dict: {track_id: [TrackedVehicle, TrackedVehicle, ...]}
        Each list is that vehicle's full trajectory across the video.
        """
        results = self.model.track(
            source=video_path,
            tracker="bytetrack.yaml",   # <-- this is the key line
            persist=True,
            device=self.device,
            verbose=False,
            stream=True,
        )

        tracks = defaultdict(list)
        for frame_idx, r in enumerate(results):
            if r.boxes.id is None:
                continue  # no tracks confirmed yet this frame
            for box, track_id, cls_idx, conf in zip(
                r.boxes.xyxy, r.boxes.id, r.boxes.cls, r.boxes.conf
            ):
                cls_name = self.model.names[int(cls_idx)]
                if cls_name not in self.VEHICLE_CLASSES:
                    continue
                tracks[int(track_id)].append(TrackedVehicle(
                    track_id=int(track_id),
                    class_name=cls_name,
                    bbox=box.tolist(),
                    frame_idx=frame_idx,
                    confidence=float(conf),
                ))
        return dict(tracks)