# Smart Traffic Monitoring — V-JEPA Powered Anomaly Detection

A self-supervised urban traffic anomaly detection and predictive monitoring framework. Uses Meta's V-JEPA 2 as a frozen feature extractor, paired with vehicle detection, multi-object tracking, and temporal prediction, to flag unusual traffic behavior — without manual labeling of "anomaly" events.

**Status:** CV pipeline (detection → tracking → feature extraction) is implemented and verified on real traffic footage. LSTM prediction, backend, and dashboard are in progress.

---

## Architecture

```
Traffic video
   → YOLOv8 detects vehicles per frame
   → ByteTrack links detections into per-vehicle tracks
   → V-JEPA 2.0 (ViT-L) extracts a feature vector per tracked vehicle
   → LSTM predicts expected future motion              [in progress]
   → Anomaly scorer flags deviations from prediction     [in progress]
   → FastAPI + PostgreSQL store and serve events         [in progress]
   → React dashboard displays live feed + alerts         [in progress]
```

## Tech stack

| Layer | Tools |
|---|---|
| Detection | YOLOv8 (Ultralytics) |
| Tracking | ByteTrack (via Ultralytics built-in tracker) |
| Feature extraction | V-JEPA 2.0 ViT-L (frozen, via HuggingFace `transformers`) |
| Prediction | LSTM (PyTorch) |
| Backend | FastAPI |
| Database | PostgreSQL |
| Frontend | React |
| Environment | Anaconda, Python 3.11, PyTorch 2.7.1 + CUDA 11.8 |

## Project structure

```
smart-traffic/
├── vision/           # detection, tracking, feature extraction (Member 1)
│   ├── detector.py   # YOLOv8 wrapper
│   ├── tracker.py    # ByteTrack wrapper
│   ├── extractor.py  # V-JEPA feature extractor
│   └── pipeline.py   # full CV pipeline: video -> per-vehicle features
├── ml/               # LSTM + anomaly detection (Member 2)
├── backend/          # FastAPI app, DB models, routes (Member 3)
├── dashboard/         # React frontend (Member 3)
├── tests/            # smoke tests for each module
├── notebooks/        # exploratory / Colab work
└── requirements.txt
```

---

## Setup guide

### 1. Prerequisites

- Windows with [Anaconda](https://www.anaconda.com/download) installed
- NVIDIA GPU with CUDA support (tested on GTX 1650, 4GB VRAM)
- Git

> **Important:** always use **Anaconda Prompt**, not plain CMD or PowerShell, for environment setup. Conda is not added to the system PATH by default.

### 2. Clone the repo

```bash
git clone https://github.com/Adarsh041-arch/Smart_traffic_monitoring_VJEPA.git
cd Smart_traffic_monitoring_VJEPA
```

### 3. Create the conda environment

```bash
conda create -n traffic python=3.11 -y
conda activate traffic
```

Your prompt should now show `(traffic)` at the start of the line.

> If you hit `LockError: Failed to acquire lock`, run `conda clean --locks` and retry.

### 4. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is detected:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

This should print `True`. If it prints `False`, double check you're running from Anaconda Prompt with `(traffic)` active — and run `where python` to confirm it's pointing at `...\anaconda3\envs\traffic\python.exe`, not a system Python install.

### 5. Install remaining dependencies

```bash
pip install ultralytics opencv-python-headless fastapi uvicorn sqlalchemy psycopg2-binary einops timm numpy pandas transformers torchcodec
```

### 6. Verify the setup

Run the smoke tests in order:

```bash
python tests/test_detector.py     # YOLOv8 detection
python tests/test_tracker.py      # ByteTrack tracking
python tests/test_extractor.py    # V-JEPA feature extraction
python tests/test_pipeline.py     # full pipeline, end-to-end
```

Each should print `Smoke test passed` (or `Pipeline smoke test passed` for the last one). The first run of `test_extractor.py` / `test_pipeline.py` downloads V-JEPA weights (~1.2GB) from HuggingFace — this can take several minutes depending on your connection.

> **Note:** to avoid HuggingFace Hub rate limits, set an `HF_TOKEN` environment variable once you have a free HuggingFace account.

### 7. Hardware notes

- YOLOv8n uses ~1.1GB VRAM — safe on 4GB cards.
- V-JEPA 2.0 ViT-L uses ~3–3.5GB VRAM in fp16 — fits on a 4GB card, but keep batch size at 1 and avoid running other GPU processes simultaneously.
- If you hit out-of-memory errors, add `torch.cuda.empty_cache()` between calls, or fall back to CPU with `device="cpu"` in the relevant module's constructor.

---

## Team

| Member | Owns |
|---|---|
| Member 1 | CV pipeline — detection, tracking, feature extraction |
| Member 2 | LSTM prediction, anomaly scoring, evaluation |
| Member 3 | FastAPI backend, PostgreSQL schema, React dashboard |

API contract is defined by Member 3 by end of Week 1 so all three branches can build independently and integrate later.

## Known limitations

- ByteTrack can lose a vehicle's identity during occlusion and reassign a new track ID when it reappears (ID switch) — a known limitation of tracking-by-detection approaches, not specific to this project.
- Some vehicle types without a native YOLO class (e.g. auto-rickshaws) get approximated as the nearest available class (car/truck) at lower confidence.