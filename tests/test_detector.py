from vision.detector import VehicleDetector
import urllib.request, os

def test_detector_smoke():
    # download a small test clip if not present
    url = "https://ultralytics.com/assets/decelera_landscape.mp4"
    path = "tests/sample.mp4"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    det = VehicleDetector()
    results = det.detect_video(path)
    assert len(results) > 0, "No detections found"
    assert all(d.class_name in det.VEHICLE_CLASSES for d in results)
    print(f"Detected {len(results)} vehicles across all frames")

if __name__ == "__main__":
    test_detector_smoke()
    print("Smoke test passed")