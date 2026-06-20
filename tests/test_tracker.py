import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vision.tracker import VehicleTracker

def test_tracker_smoke():
    tracker = VehicleTracker()
    tracks = tracker.track_video("tests/traffic_real.mp4")

    print(f"Total unique vehicles tracked: {len(tracks)}")
    for tid, history in list(tracks.items())[:5]:
        print(f"  Track {tid}: {history[0].class_name}, "
              f"{len(history)} frames, "
              f"frames {history[0].frame_idx}-{history[-1].frame_idx}")

    assert len(tracks) > 0, "No tracks found"
    print("Smoke test passed")

if __name__ == "__main__":
    test_tracker_smoke()