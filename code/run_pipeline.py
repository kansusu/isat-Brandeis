"""
Python equivalent of bash-test-1118.sh.
Runs the 5-step pipeline using existing step scripts.
"""
import os
import sys
import time
import subprocess
import shutil
from typing import Dict, Any
from dotenv import dotenv_values

import torch

def log_device(label: str):
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[{label}] Using GPU: {name}")
    else:
        print(f"[{label}] Using CPU (CUDA not available)")


REQUIRED_KEYS = [
    "FRAME_FOLDER",
    "FRAME_SKIP_RATE",
    "STEP2_VISUALS",
    "STEP3_GAZE_VISUALS",
    "HEADBOX_JSON",
    "POSTURE_JSON",
    "OBJECT_CSV",
    "MMDET_CONFIG",
    "MMDET_CHECKPOINT",
    "GAZELLE_CHECKPOINT",
    "STEP2_MODE",
]


def load_config(config_path: str = "config.env") -> Dict[str, Any]:
    cfg = dotenv_values(config_path)
    missing = [k for k in REQUIRED_KEYS if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise SystemExit(f"Missing required keys in {config_path}: {', '.join(missing)}")
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ("FRAME_FOLDER", "STEP2_VISUALS", "STEP3_GAZE_VISUALS"):
        path = cfg.get(key)
        if path:
            os.makedirs(path, exist_ok=True)


def run_step(cmd, label: str) -> float:
    print(f"\n--- [{label}] ---")
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    print(f"[{label}] done in {elapsed:.1f}s")
    return elapsed


def run_pipeline(input_video: str, config_path: str = "config.env") -> Dict[str, Any]:
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cfg = load_config(config_path)
    ensure_dirs(cfg)

    # CLEAN UP POTENTIAL DIRECTORY PATHS
    import shutil
    for json_path in [cfg["HEADBOX_JSON"], cfg["POSTURE_JSON"]]:
        if os.path.exists(json_path) and os.path.isdir(json_path):
            print(f"Removing directory that should be file: {json_path}")
            shutil.rmtree(json_path)
    
    for csv_path in [cfg["OBJECT_CSV"]]:
        if os.path.exists(csv_path) and os.path.isdir(csv_path):
            print(f"Removing directory that should be file: {csv_path}")
            shutil.rmtree(csv_path)
    
    totals = {}
    
    log_device("Step 1 setup")

    print(f"----Frame Skip Rate: {cfg['FRAME_SKIP_RATE']}----")

    totals["step1"] = run_step([
        sys.executable, "step1_extract.py",
        input_video,
        cfg["FRAME_FOLDER"],
        str(cfg["FRAME_SKIP_RATE"])
    ], "Step 1: Extract Frames")

    totals["step2"] = run_step([
        sys.executable, "step2_postures_headboxes.py",
        "--input_folder", cfg["FRAME_FOLDER"],
        "--output_headbox_json", cfg["HEADBOX_JSON"],
        "--output_visual_folder", cfg["STEP2_VISUALS"],
        "--output_posture_json", cfg["POSTURE_JSON"],
        "--mode", cfg["STEP2_MODE"],
    ], "Step 2: Posture/Headboxes")

    totals["step3"] = run_step([
        sys.executable, "step3_gaze.py",
        "--gazelle_checkpoint", cfg["GAZELLE_CHECKPOINT"],
        "--image_folder", cfg["FRAME_FOLDER"],
        "--head_box_json", cfg["HEADBOX_JSON"],
        "--posture_json", cfg["POSTURE_JSON"],
        "--output_dir", cfg["STEP3_GAZE_VISUALS"],
    ], "Step 3: Gaze Inference")

    totals["step4"] = run_step([
        sys.executable, "step4_obj_detection.py",
        cfg["FRAME_FOLDER"],
        cfg["MMDET_CONFIG"],
        cfg["MMDET_CHECKPOINT"],
        cfg["OBJECT_CSV"],
    ], "Step 4: Object Detection")

    totals["step5"] = run_step([
        sys.executable, "step5_gaze_obj_association.py",
        cfg["POSTURE_JSON"],
        cfg["OBJECT_CSV"],
        cfg["SUMMARY_JSON"],
    ], "Step 5: Gaze/Object Association")

    total_time = sum(totals.values())
    print("\n=== Pipeline complete ===")
    for k, v in totals.items():
        print(f"{k}: {v:.1f}s")
    print(f"Total: {total_time:.1f}s")

    return {
        "config": cfg,
        "timings": totals,
        "total_time": total_time,
        "outputs": {
            "frames": cfg["FRAME_FOLDER"],
            "step2_visuals": cfg["STEP2_VISUALS"],
            "step3_visuals": cfg["STEP3_GAZE_VISUALS"],
            "headboxes": cfg["HEADBOX_JSON"],
            "posture": cfg["POSTURE_JSON"],
            "objects": cfg["OBJECT_CSV"],
            "summary": cfg["SUMMARY_JSON"],
        },
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <input_video_path> [config_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.env"
    run_pipeline(video_path, config_path)


if __name__ == "__main__":
    main()
