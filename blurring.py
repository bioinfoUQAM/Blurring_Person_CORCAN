#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-GPU person/head detection, tracking, gap-filling, and anonymization.

- Spawns one worker per GPU (up to 4). Each worker processes a subset of videos.
- YOLO (Ultralytics) for detection; DeepSORT for tracking (TorchReID/OSNet preferred, Mobilenet fallback).
- Stores per-track boxes and fills short gaps via linear interpolation.
- Applies anonymization (pixelation by default) to detected and interpolated boxes.
- Mirrors INPUT_DIR's folder structure inside OUTPUT_DIR.
- Boxes/labels overlay disabled by default (DRAW_BOXES=False).
"""

import os
import sys
import cv2
import importlib
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

# Imports that do not create CUDA context in the parent
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# =========================
# CONFIGURATIONS
# =========================
MODEL_PATH = "/mnt/4/best.pt"
INPUT_DIR  = "/mnt/4/last"  # input folder with videos (recursively)
OUTPUT_DIR = "/mnt/4/lastfinal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Online tracking (DeepSORT) settings
FPS_FALLBACK  = 30     # used if the input video does not report FPS
GAP_SECONDS   = 6.0    # track can survive up to this number of seconds without a fresh detection
N_INIT        = 6      # frames required to confirm a new track (higher reduces false tracks)
MAX_COSINE    = 0.30   # stricter ReID distance threshold (lower is stricter)
NN_BUDGET     = 1024   # gallery size for embeddings

# Hysteresis-style entry thresholds for detections (per class)
ENTER_TH      = {"person": 0.40, "head": 0.20}
DEFAULT_ENTER = 0.50   # fallback threshold if class name is missing

# Only commit a track update if there was a match (prevents "walking boxes")
RECENT_MATCH_MAX = 0   # 0 means the track must have matched in the current frame

# Offline interpolation settings (applied after the first pass)
INTERP_MAX_SECONDS = 4.0  # max length of gaps to fill per track, in seconds
MIN_TRACK_LEN      = 6    # minimum number of detected frames required to interpolate a track
MIN_BOX_AREA       = 12 * 12  # ignore very small boxes as noise

# Anonymization settings
BLUR_MODE         = "pixel"  # "pixel" or "gauss"
BLUR_MARGIN_RATIO = 0.08     # expand the anonymized ROI by a margin to avoid edge leaks
BLUR_MIN_KERNEL   = 51       # min kernel size for Gaussian blur (odd)
BLUR_SCALE        = 0.25     # Gaussian kernel as a fraction of the ROI's min side
PIXEL_BLOCK       = 15       # pixelation block size (bigger = coarser)
DRAW_BOXES        = False    # overlay boxes/labels on top of anonymization

# Colors for optional overlay (unused when DRAW_BOXES=False)
COLOR_PERSON = (0, 255, 0)
COLOR_HEAD   = (255, 0, 0)
COLOR_INTERP = (0, 255, 255)

# =========================
# HELPERS
# =========================
def resolve_embedder():
    """
    Try to use TorchReID (OSNet). If the module is not available,
    fall back to Mobilenet (TensorFlow) on CPU to avoid GPU conflicts.
    """
    try:
        import torchreid  # noqa
        importlib.import_module("torchreid.utils")
        return "torchreid"
    except Exception:
        return "mobilenet"

def make_tracker(max_age, device, embedder):
    """
    Build a DeepSORT tracker. If TorchReID is available, use it on the selected device.
    Otherwise, use Mobilenet embedder on CPU to avoid GPU resource conflicts with PyTorch.
    """
    gpu = ("cuda" in device)
    if embedder == "torchreid":
        try:
            return DeepSort(
                max_age=max_age,
                n_init=N_INIT,
                max_cosine_distance=MAX_COSINE,
                nn_budget=NN_BUDGET,
                nms_max_overlap=1.0,
                embedder="torchreid",
                half=True,
                bgr=True,
                embedder_gpu=gpu,
            )
        except Exception as e:
            print(f"[WARN] TorchReID failed, falling back to Mobilenet: {e}")

    return DeepSort(
        max_age=max_age,
        n_init=N_INIT,
        max_cosine_distance=MAX_COSINE,
        nn_budget=NN_BUDGET,
        nms_max_overlap=1.0,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=False,  # run TF mobilenet on CPU
    )

def odd(n: int) -> int:
    """Ensure the integer is odd (required by Gaussian kernel)."""
    return n if n % 2 == 1 else n + 1

def strong_anonymize_roi(frame, box_xyxy):
    """
    Apply anonymization to the ROI (pixelation by default, or Gaussian blur).
    Expands the box by a margin to reduce boundary leakage.
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box_xyxy)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return

    # Expand the ROI by a margin proportional to box size
    mx, my = int(w * BLUR_MARGIN_RATIO), int(h * BLUR_MARGIN_RATIO)
    X1, Y1 = max(0, x1 - mx), max(0, y1 - my)
    X2, Y2 = min(W, x2 + mx), min(H, y2 + my)
    roi = frame[Y1:Y2, X1:X2]
    if roi.size <= 0:
        return

    if BLUR_MODE == "gauss":
        # Gaussian blur with kernel based on ROI size
        k = max(BLUR_MIN_KERNEL, int(min(X2 - X1, Y2 - Y1) * BLUR_SCALE))
        k = odd(max(3, k))
        frame[Y1:Y2, X1:X2] = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        # Pixelation: downscale then upscale with nearest-neighbor
        rh, rw = roi.shape[:2]
        pw = max(1, rw // PIXEL_BLOCK)
        ph = max(1, rh // PIXEL_BLOCK)
        small = cv2.resize(roi, (pw, ph), interpolation=cv2.INTER_LINEAR)
        pixel = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        frame[Y1:Y2, X1:X2] = pixel

def draw_box(img, box, text, color, thick=2):
    """
    Optional overlay for debugging or review. Disabled globally by DRAW_BOXES.
    """
    if not DRAW_BOXES:
        return
    x1, y1, x2, y2 = map(int, box)
    if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    cv2.putText(img, text, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def interpolate_series(series_dict, max_gap_frames):
    """
    For each track ID, fill short gaps between detections by linearly interpolating boxes.

    Args:
        series_dict: dict[track_id] -> dict[frame_idx] -> [x1, y1, x2, y2]
        max_gap_frames: maximum number of frames to fill between two real detections

    Returns:
        dict[track_id] -> dict[frame_idx] -> (box, source)
        where source ∈ {"det", "interp"} to distinguish real vs interpolated boxes.
    """
    out = {}
    for tid, fmap in series_dict.items():
        # Skip very short tracks (not enough signal to interpolate reliably)
        if len(fmap) < MIN_TRACK_LEN:
            out[tid] = {fi: (box, "det") for fi, box in fmap.items()}
            continue

        frames_sorted = sorted(fmap.keys())
        out_map = {}
        for i, f in enumerate(frames_sorted):
            out_map[f] = (fmap[f], "det")

            # If this is the last real frame in the track, nothing to interpolate after it
            if i == len(frames_sorted) - 1:
                break

            # Gap between this frame and the next detection
            fn = frames_sorted[i + 1]
            gap = fn - f - 1
            if gap <= 0 or gap > max_gap_frames:
                continue

            # Linear interpolation between the two boxes
            b1, b2 = fmap[f], fmap[fn]
            x1a, y1a, x2a, y2a = b1
            x1b, y1b, x2b, y2b = b2
            for g in range(1, gap + 1):
                t = g / (gap + 1.0)
                x1 = int(round(x1a * (1 - t) + x1b * t))
                y1 = int(round(y1a * (1 - t) + y1b * t))
                x2 = int(round(x2a * (1 - t) + x2b * t))
                y2 = int(round(y2a * (1 - t) + y2b * t))
                out_map[f + g] = ([x1, y1, x2, y2], "interp")

        out[tid] = out_map
    return out

# =========================
# WORKER (ONE PER GPU)
# =========================
def worker_process(video_list, gpu_id):
    """
    Worker process:
      - isolates a single GPU via CUDA_VISIBLE_DEVICES
      - loads YOLO onto that GPU
      - processes its share of videos
      - mirrors INPUT_DIR subfolders inside OUTPUT_DIR
    """
    # Select a single visible GPU for this process BEFORE any CUDA calls
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[PROC->GPU{gpu_id}] visible device: {device}")

    # Load the model on the selected device
    model = YOLO(MODEL_PATH).to(device)
    names = model.model.names
    embedder = resolve_embedder()

    for video_path in video_list:
        try:
            # Mirror folder structure from INPUT_DIR inside OUTPUT_DIR
            rel_path = os.path.relpath(video_path, INPUT_DIR)   # e.g., "sub1/sub2/video.mp4"
            rel_dir  = os.path.dirname(rel_path)                 # e.g., "sub1/sub2"
            out_dir  = os.path.join(OUTPUT_DIR, rel_dir)         # e.g., ".../OUTPUT_DIR/sub1/sub2"
            os.makedirs(out_dir, exist_ok=True)

            stem    = Path(video_path).stem
            out_mp4 = os.path.join(out_dir, f"{stem}_blurred.mp4")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[GPU{gpu_id}] ERROR opening: {video_path}")
                continue

            w, h = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
            max_age = int(GAP_SECONDS * fps)

            # Build one tracker for person and one for head
            tracker_person = make_tracker(max_age, device, embedder)
            tracker_head   = make_tracker(max_age, device, embedder)

            # Temporary store of per-track boxes per frame (only confirmed matches)
            persons_series, heads_series = defaultdict(dict), defaultdict(dict)

            fidx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # YOLO inference
                res = model(frame, verbose=False)[0]

                # Gather detections that meet the entry thresholds and min box size
                detP, detH = [], []
                if res.boxes is not None:
                    for b in res.boxes:
                        cid = int(b.cls[0])
                        cname = str(names[cid] if not isinstance(names, dict) else names.get(cid)).lower()
                        if cname not in ("person", "head"):
                            continue
                        conf = float(b.conf[0])
                        if conf < ENTER_TH.get(cname, DEFAULT_ENTER):
                            continue
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                            continue
                        if cname == "person":
                            detP.append(([x1, y1, x2 - x1, y2 - y1], conf, cname))
                        else:
                            detH.append(([x1, y1, x2 - x1, y2 - y1], conf, cname))

                # DeepSORT updates (commit only if matched in this frame)
                for t in tracker_person.update_tracks(detP, frame=frame):
                    if t.is_confirmed() and getattr(t, "time_since_update", 0) <= RECENT_MATCH_MAX:
                        persons_series[int(t.track_id)][fidx] = list(map(int, t.to_ltrb()))
                for t in tracker_head.update_tracks(detH, frame=frame):
                    if t.is_confirmed() and getattr(t, "time_since_update", 0) <= RECENT_MATCH_MAX:
                        heads_series[int(t.track_id)][fidx] = list(map(int, t.to_ltrb()))

                fidx += 1

            cap.release()

            # Offline: fill gaps by interpolation
            max_gap_frames = int(INTERP_MAX_SECONDS * fps)
            persons_full = interpolate_series(persons_series, max_gap_frames)
            heads_full   = interpolate_series(heads_series,   max_gap_frames)

            # Render the final anonymized output
            cap2 = cv2.VideoCapture(video_path)
            writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            fi = 0
            while True:
                ok, frame = cap2.read()
                if not ok:
                    break

                # Anonymize all boxes (both detected and interpolated)
                for tid, fmap in persons_full.items():
                    if fi in fmap:
                        box, _src = fmap[fi]
                        strong_anonymize_roi(frame, box)
                for tid, fmap in heads_full.items():
                    if fi in fmap:
                        box, _src = fmap[fi]
                        strong_anonymize_roi(frame, box)

                # Optional overlays (disabled by default)
                for tid, fmap in persons_full.items():
                    if fi in fmap:
                        box, src = fmap[fi]
                        draw_box(frame, box,
                                 f"person#{tid}{'' if src=='det' else ' [interp]'}",
                                 COLOR_PERSON if src == "det" else COLOR_INTERP)
                for tid, fmap in heads_full.items():
                    if fi in fmap:
                        box, src = fmap[fi]
                        draw_box(frame, box,
                                 f"head#{tid}{'' if src=='det' else ' [interp]'}",
                                 COLOR_HEAD if src == "det" else COLOR_INTERP)

                writer.write(frame)
                fi += 1

            cap2.release()
            writer.release()
            print(f"[GPU{gpu_id}] Saved: {out_mp4}")

        except Exception as e:
            print(f"[GPU{gpu_id}] Failed on {video_path}: {e}")

# =========================
# MAIN (spawn)
# =========================
def chunk_round_robin(items, n):
    """
    Split items into n lists using round-robin:
    [A, B, C, D, A, B, C, D, ...]
    """
    chunks = [[] for _ in range(n)]
    for i, it in enumerate(items):
        chunks[i % n].append(it)
    return chunks

if __name__ == "__main__":
    # Required for CUDA + multiprocessing on Linux
    mp.set_start_method("spawn", force=True)

    # Collect videos from INPUT_DIR (recursive)
    videos = []
    for r, _, fs in os.walk(INPUT_DIR):
        for f in fs:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                videos.append(os.path.join(r, f))
    if not videos:
        print("No videos found in:", INPUT_DIR)
        sys.exit(0)

    # Discover GPUs and cap at 4 (one worker per GPU as requested)
    avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
    NUM_GPUS = min(4, avail) if avail > 0 else 0
    if NUM_GPUS <= 0:
        print("No GPU detected. Aborting.")
        sys.exit(1)

    # Distribute videos across GPUs and launch a worker per GPU
    parts = chunk_round_robin(videos, NUM_GPUS)
    ctx = mp.get_context("spawn")
    procs = []
    for gid in range(NUM_GPUS):
        if not parts[gid]:
            continue
        p = ctx.Process(target=worker_process, args=(parts[gid], gid), daemon=False)
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("Done.")
