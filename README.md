# Results of blurring in many cameras!

<img width="927" height="878" alt="Captura de Tela 2025-08-28 às 2 06 59 AM" src="https://github.com/user-attachments/assets/0998bb4c-155b-4bb2-9cb7-48ad156ca23d" />
<img width="1642" height="855" alt="Captura de Tela 2025-08-28 às 2 07 16 AM" src="https://github.com/user-attachments/assets/20b5d673-4d97-4fb9-ad30-d825998f5668" />
<img width="1702" height="849" alt="Captura de Tela 2025-08-28 às 2 07 38 AM" src="https://github.com/user-attachments/assets/5ddc8e52-975f-41a7-8da2-6631d8577cce" />
<img width="1625" height="828" alt="Captura de Tela 2025-08-28 às 2 07 50 AM" src="https://github.com/user-attachments/assets/3a8ad549-efe9-41cd-9130-efab34b80736" />


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


# Update Compute Canada:

We are reaching out to inform you that the Cedar compute cluster will be retired on September 12, 2025.

Data Access

Files stored on Cedar are already available on Fir because the two clusters share the same file systems; no action is required regarding your stored files.

Starting September 12, please submit your jobs to Fir or another cluster on our new national infrastructure.
