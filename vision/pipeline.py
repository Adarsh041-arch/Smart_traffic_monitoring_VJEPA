import cv2
import numpy as np
from vision.tracker import VehicleTracker
from vision.extractor import VJEPAFeatureExtractor


class TrafficPipeline:
    def __init__(self, min_track_len=16, clip_len=64):
        self.tracker = VehicleTracker()
        self.extractor = VJEPAFeatureExtractor()
        self.min_track_len = min_track_len
        self.clip_len = clip_len

    def _pad_or_sample(self, crops):
        """Make exactly clip_len frames — sample if too many, pad if too few."""
        n = len(crops)
        if n >= self.clip_len:
            idxs = np.linspace(0, n - 1, self.clip_len).astype(int)
            return np.array([crops[i] for i in idxs])
        crops = crops + [crops[-1]] * (self.clip_len - n)
        return np.array(crops)

    def process(self, video_path: str):
        """Returns {track_id: {'class': str, 'features': tensor, 'n_frames': int}}"""

        # 1. Run detection + tracking once
        tracks = self.tracker.track_video(video_path)

        # 2. Drop short/noisy tracks before doing any expensive work
        valid_tracks = {
            tid: history for tid, history in tracks.items()
            if len(history) >= self.min_track_len
        }
        if not valid_tracks:
            return {}

        # 3. Build a frame_idx -> [(track_id, bbox)] lookup
        frame_to_boxes = {}
        for tid, history in valid_tracks.items():
            for h in history:
                frame_to_boxes.setdefault(h.frame_idx, []).append((tid, h.bbox))

        # 4. Single pass through the video — crop every track's box per frame
        track_crops = {tid: [] for tid in valid_tracks}
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_to_boxes:
                for tid, bbox in frame_to_boxes[frame_idx]:
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        crop = cv2.resize(crop, (256, 256))
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        track_crops[tid].append(crop)
            frame_idx += 1
        cap.release()

        # 5. Run V-JEPA once per track (only step that must stay sequential)
        results = {}
        for tid, crops in track_crops.items():
            if len(crops) == 0:
                continue
            clip = self._pad_or_sample(crops)
            features = self.extractor.extract_clip(clip)
            results[tid] = {
                "class": valid_tracks[tid][0].class_name,
                "features": features,
                "n_frames": len(valid_tracks[tid]),
            }
        return results