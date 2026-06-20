import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from vision.extractor import VJEPAFeatureExtractor


def make_synthetic_video(path, num_frames=90, size=(320, 240)):
    """Creates a small synthetic video locally — no internet needed."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 20, size)
    for i in range(num_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        x = int((i / num_frames) * size[0])
        cv2.rectangle(frame, (x, 100), (x + 40, 140), (0, 200, 0), -1)
        writer.write(frame)
    writer.release()


def load_frames(video_path, num_frames=64):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = cv2.resize(frame, (256, 256))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.array(frames[:num_frames])


def test_extractor_smoke():
    path = "tests/traffic_real.mp4"
    if not os.path.exists(path):
        make_synthetic_video(path)

    print("Loading frames...")
    frames = load_frames(path)
    print(f"Frame batch shape: {frames.shape}")

    print("Loading V-JEPA extractor...")
    extractor = VJEPAFeatureExtractor()

    print("Running inference...")
    features = extractor.extract_clip(frames)
    print(f"Feature vector shape: {features.shape}")

    assert features.shape[0] == 1
    print("Smoke test passed")


if __name__ == "__main__":
    test_extractor_smoke()