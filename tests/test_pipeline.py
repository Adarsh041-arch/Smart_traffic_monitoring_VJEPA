import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vision.pipeline import TrafficPipeline

def test_pipeline_smoke():
    pipeline = TrafficPipeline(min_track_len=16)
    results = pipeline.process("tests/traffic_real.mp4")

    print(f"Vehicles with extracted features: {len(results)}")
    for tid, data in list(results.items())[:5]:
        print(f"  Track {tid}: {data['class']}, "
              f"{data['n_frames']} frames, "
              f"feature shape {data['features'].shape}")

    assert len(results) > 0
    print("Pipeline smoke test passed")

if __name__ == "__main__":
    test_pipeline_smoke()