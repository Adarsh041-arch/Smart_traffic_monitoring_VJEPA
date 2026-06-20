# tests/test_real_detection.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vision.detector import VehicleDetector

det = VehicleDetector()
results = det.detect_video("tests/traffic_real.mp4")
print(f"Total detections: {len(results)}")
for d in results[:10]:
    print(d.class_name, d.confidence, d.bbox)