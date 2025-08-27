# What it does

Detection (YOLOv8): Runs Ultralytics YOLO on each frame, filtering to person and head with per-class confidence thresholds.

Tracking (DeepSORT): Maintains stable IDs via appearance embeddings:

Prefers TorchReID (OSNet) for stronger ReID; falls back to Mobilenet on CPU if TorchReID isn’t available.

Gap filling (offline interpolation): After the first pass, linearly interpolates missing boxes for each track up to INTERP_MAX_SECONDS (e.g., through short occlusions).

Anonymization: Applies pixelation (default) or Gaussian blur to every box (detected + interpolated) with a configurable safety margin to avoid edge leaks.

Multi-GPU execution: Spawns one process per GPU (up to 4), each handling a subset of videos. The model is loaded inside the worker after selecting the GPU (CUDA-safe).

No overlays by default: Bounding boxes/labels are disabled (DRAW_BOXES=False), so the output is just anonymized video.

# Why it’s robust

Hysteresis-like thresholds (ENTER_TH) to start tracks only from confident detections.

Tracker survival window (GAP_SECONDS) to keep IDs alive through short misses.

Strict ReID matching (MAX_COSINE) to reduce ID switches.

Interpolation to “fill” brief gaps (e.g., 3–4 seconds) so anonymization doesn’t flicker.

# Requirements

Python 3.9+

PyTorch + CUDA (for GPU)

ultralytics (YOLOv8)

deep_sort_realtime

opencv-python

pip install ultralytics deep-sort-realtime opencv-python

pip install torchreid
# Usage
Edit the paths at the top of the script:

MODEL_PATH: YOLO weights (e.g., /path/to/best.pt)

INPUT_DIR: folder with input videos

OUTPUT_DIR: where anonymized videos will be saved

# Limitations

Linear interpolation assumes smooth motion; very long occlusions won’t be reconstructed.

Best performance with stable FPS and well-trained YOLO weights that include person and head classes.

# Additional informations 
The model training was conducted on the Compute Canada Cedar cluster using the dataset located at:

/scratch/Blurring2025/Dataset_4_head_person/

For pre-training, we utilized the CrowdHuman dataset available at:

/scratch/Blurring2025/crowdhuman/

Subsequently, fine-tuning was performed using our custom dataset stored in Dataset_4_head_person.

We trained using yolov8, and you can run using the train_yolov8.sh if necessary.
