# infer_boudingboxandarrow.py (updated)
# Gaze-only inference.
# - BLUE: mutual gaze
# - GREEN: joint attention
# - RED: default (including "look_at_other")

import os
import json
from PIL import Image
import torch
import numpy as np
import sys 
import time 
import argparse 
import gc
import torch 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "gazelle"))

from gazelle.model import get_gazelle_model
from gazelle.utils import visualize_gaze_arrows
from run_pipeline import log_device

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE — expected GPU")

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"[GPU] Using {torch.cuda.get_device_name(0)}")


# === Config ===
# 🟥 --- These are now DEFAULTS, can be overridden by command line --- 🟥
DEFAULT_IMAGE_FOLDER = "/localcache/zhuyifan/Bodyskeleton/test-1108-10s"
DEFAULT_HEAD_BOX_JSON = "/localcache/zhuyifan/Bodyskeleton/headboxes_from_test-1108-10s.json"
DEFAULT_POSTURE_JSON_PATH = "/localcache/zhuyifan/Bodyskeleton/posture_test-1108-10s.json"
DEFAULT_GAZELLE_CHECKPOINT = "/home/ubuntu/multimodal-attention-analyzer/gazelle_dinov2_vitl14_inout.pt"
DEFAULT_OUTPUT_DIR = "gaze_predictions_164"

# --- Tunable thresholds ---
INOUT_THR = 0.5            
EXPAND_PAD_PX = 18         
EXPAND_PAD_RATIO = 0.18    
NEAR_EDGE_PX = 16          
FINE_CLUSTER_FRAC = 0.03   
LOOSE_CLUSTER_FRAC = 0.08  

torch.cuda.empty_cache()
gc.collect()
torch.cuda.synchronize()

# === Load Gaze Model (Gazelle) ===
def setup_gazelle(checkpoint_path):
    print("[INFO] Initializing gaze model (Gazelle)...")
    model, transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Gazelle checkpoint not found at: {checkpoint_path}")
        sys.exit(1)
    state = torch.load(checkpoint_path, weights_only=True)
    # model.load_gazelle_state_dict(state)
    # model.eval()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    model.load_gazelle_state_dict(state)
    model.eval()

    device = torch.device("cuda")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    model.to(device)
    print("Model dtype:", next(model.parameters()).dtype)
    log_device(f"Step 3 gaze -> {device}")
    print(f"[INFO] Gaze model initialized on {device}.")
    return model, transform, device

# === Load Head Boxes ===
def load_json_data(json_path, name="Data"):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"[INFO] Loaded {name} from {json_path}")
        return data
    except FileNotFoundError:
        print(f"[ERROR] Could not find {name} JSON at {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] {name} JSON at {json_path} is corrupted.")
        sys.exit(1)

# ---------- small geometry helpers ----------
def expand_bbox(bx, W, H, pad_px=EXPAND_PAD_PX, pad_ratio=EXPAND_PAD_RATIO):
    x1, y1, x2, y2 = bx
    w, h = x2 - x1, y2 - y1
    p = max(pad_px, int(max(w, h) * pad_ratio))
    nx1 = max(0, x1 - p); ny1 = max(0, y1 - p)
    nx2 = min(W, x2 + p); ny2 = min(H, y2 + p)
    return (nx1, ny1, nx2, ny2)

def point_in_bbox_or_near(pt, bx, near_px=NEAR_EDGE_PX):
    x, y = pt
    x1, y1, x2, y2 = bx
    if x1 <= x <= x2 and y1 <= y <= y2:
        return True
    # near-edge tolerance
    nx1, ny1, nx2, ny2 = x1 - near_px, y1 - near_px, x2 + near_px, y2 + near_px
    return (nx1 <= x <= nx2) and (ny1 <= y <= ny2)

def seg_intersects_bbox(p0, p1, bx):
    # Liang–Barsky segment-rectangle clipping
    x1, y1, x2, y2 = bx
    x0, y0 = p0; x1s, y1s = p1
    dx, dy = x1s - x0, y1s - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - x1, x2 - x0, y0 - y1, y2 - y0]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = -qi / pi
            if pi < 0:
                if t > u1: return False
                if t > u0: u0 = t
            else:
                if t < u0: return False
                if t < u1: u1 = t
    return True

def head_center(bx):
    x1, y1, x2, y2 = bx
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# 🟥 --- MAIN FUNCTION --- 🟥
def main(args):
    # === Load Models ===
    model, transform, device = setup_gazelle(args.gazelle_checkpoint)
    
    # === Load Data ===
    head_boxes = load_json_data(args.head_box_json, "Head Box")
    posture_data = load_json_data(args.posture_json, "Posture")

    # === Ensure output dir ===
    os.makedirs(args.output_dir, exist_ok=True)

    # === Iterate through images ===
    print(f"[INFO] Starting gaze processing for folder: {args.image_folder}") 
    total_start_time = time.time() 
    frame_count = 0 

    image_files = sorted([f for f in os.listdir(args.image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    for img_name in image_files:
        img_path = os.path.join(args.image_folder, img_name)
        try:
            image_pil = Image.open(img_path).convert("RGB") # For Gazelle
        except Exception as e:
            print(f"[Warning] Could not open image {img_path}: {e}. Skipping.")
            continue

        if img_name not in head_boxes or not head_boxes[img_name]:
            print(f"[Warning] No head boxes for {img_name}. Skipping.")
            continue
            
        # 🟥 --- START: CRITICAL FIX FOR HEADBOX_ONLY MODE --- 🟥
        if img_name not in posture_data:
            # If the frame exists in headboxes but not in posture data (e.g., from headbox_only mode in Step 2)
            posture_data[img_name] = {}
            print(f"[INFO] Frame {img_name} missing from posture data. Initializing empty entry.")
        # 🟥 --- END: CRITICAL FIX FOR HEADBOX_ONLY MODE --- 🟥

        frame_count += 1
        frame_start_time = time.time()

        boxes_dict = head_boxes[img_name]
        sorted_keys = sorted(boxes_dict.keys(), key=int)
        
        # Check for missing headboxes (e.g., None)
        bboxes_norm = []
        valid_sorted_keys = []
        for key in sorted_keys:
            if boxes_dict[key]:
                bboxes_norm.append(tuple(boxes_dict[key]))
                valid_sorted_keys.append(key)
            else:
                print(f"[Warning] Skipping person {key} in {img_name} due to missing headbox.")
        
        if not bboxes_norm:
            print(f"[Warning] No valid headboxes to process for {img_name}, skipping.")
            continue

        # Prepare input
        input_data = {
            "images": transform(image_pil).unsqueeze(0).to(device),
            "bboxes": [bboxes_norm],
        }

        # Predict Gaze
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(input_data)

        heatmaps = output["heatmap"][0].detach().cpu().numpy()   
        inout_scores = None
        if output.get("inout", None) is not None:
            inout_scores = torch.sigmoid(output["inout"][0]).detach().cpu().numpy()

        W, H = image_pil.size
        gaze_pts_px = []
        for i, hm in enumerate(heatmaps):
            arg = int(np.argmax(hm))
            y64, x64 = divmod(arg, hm.shape[1])
            gx = int((x64 + 0.5) * (W / hm.shape[1]))
            gy = int((y64 + 0.5) * (H / hm.shape[0]))
            gaze_pts_px.append((gx, gy))
            
        # ================================
        # Color rules before drawing
        # ================================
        bboxes_px = []
        for (xmin, ymin, xmax, ymax) in bboxes_norm:
            x1, y1 = int(xmin * W), int(ymin * H)
            x2, y2 = int(xmax * W), int(ymax * H)
            bboxes_px.append((x1, y1, x2, y2))

        N = len(bboxes_px)
        RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
        colors = [RED for _ in range(N)]

        valid = [True] * N
        if inout_scores is not None:
            valid = [float(inout_scores[i]) >= INOUT_THR for i in range(N)]

        attention_type = []
        attention_target_object = []
        for i in range(N):
            if not valid[i]:
                attention_type.append("out_of_frame")
                attention_target_object.append("N/A")
            else:
                attention_type.append("default") 
                attention_target_object.append("N/A")

        bboxes_expanded = [expand_bbox(bx, W, H) for bx in bboxes_px]
        head_centers = [head_center(bx) for bx in bboxes_px]

        # 1) Robust person-to-person (BLUE)
        looks_at = [set() for _ in range(N)]
        for i in range(N):
            if not valid[i]:
                continue
            p0 = head_centers[i]
            p1 = gaze_pts_px[i]
            for j in range(N):
                if i == j or not valid[j]:
                    continue
                bxj = bboxes_expanded[j]
                if point_in_bbox_or_near(p1, bxj) or seg_intersects_bbox(p0, p1, bxj):
                    looks_at[i].add(j)

        mutual_partners = [[] for _ in range(N)]
        for i in range(N):
            if not valid[i]: continue
            for j in looks_at[i]:
                if i in looks_at[j]:
                    mutual_partners[i].append(j)
        
        for i in range(N):
            if mutual_partners[i]:
                colors[i] = BLUE
                attention_type[i] = "mutual_gaze"
                target_str = ",".join(map(str, sorted(mutual_partners[i])))
                attention_target_object[i] = f"person_{target_str}"


        # 1.5) Single-person-target / "Look at Other" (RED)
        for i in range(N):
            if not valid[i]: continue
            if attention_type[i] == "mutual_gaze": continue
            if looks_at[i]:
                attention_type[i] = "look_at_other"
                target_str = ",".join(map(str, sorted(list(looks_at[i]))))
                attention_target_object[i] = f"person_{target_str}"

        # 2) Shared target (GREEN)
        diag = (W**2 + H**2) ** 0.5
        thr_fine = int(max(12, FINE_CLUSTER_FRAC * diag))   
        thr_loose = int(max(28, LOOSE_CLUSTER_FRAC * diag)) 

        adj = [[] for _ in range(N)]
        for i in range(N):
            if not valid[i] or attention_type[i] in ["mutual_gaze", "look_at_other"]:
                continue
            xi, yi = gaze_pts_px[i]
            for j in range(i + 1, N):
                if not valid[j] or attention_type[j] in ["mutual_gaze", "look_at_other"]:
                    continue
                xj, yj = gaze_pts_px[j]
                d2 = (xi - xj)**2 + (yi - yj)**2
                if d2 <= thr_fine**2 or d2 <= thr_loose**2:
                    adj[i].append(j)
                    adj[j].append(i)

        seen = [False] * N
        joint_attention_cluster_id = 0 
        for i in range(N):
            if seen[i] or not valid[i] or attention_type[i] in ["mutual_gaze", "look_at_other"]:
                continue
            if not adj[i]:
                continue
            stack = [i]; comp = []; seen[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            if len(comp) >= 2:
                
                cluster_name = f"shared_target_cluster_{joint_attention_cluster_id}"
                
                for k in comp:
                    if attention_type[k] not in ["mutual_gaze", "look_at_other"]: 
                        colors[k] = GREEN
                        attention_type[k] = "joint_attention"
                        attention_target_object[k] = cluster_name
                joint_attention_cluster_id += 1


        # --- 🟥 CRITICAL UPDATE: Write data to posture_data ---
        for i in range(N):
            person_id_str = valid_sorted_keys[i]

            # 🟥 Initialize person entry if missing (Handles 'headbox_only' mode from Step 2)
            if person_id_str not in posture_data[img_name]:
                 posture_data[img_name][person_id_str] = {} 
            
            # Add gaze attention type
            posture_data[img_name][person_id_str]["gaze_attention_type"] = attention_type[i]
            # Add gaze target (e.g., "person_1" or "shared_target_cluster_0")
            posture_data[img_name][person_id_str]["gaze_target"] = attention_target_object[i]
            # Add the raw gaze coordinates, which Step 5 needs
            posture_data[img_name][person_id_str]["gaze_x"] = int(gaze_pts_px[i][0])
            posture_data[img_name][person_id_str]["gaze_y"] = int(gaze_pts_px[i][1])
        # --- END OF CRITICAL UPDATE ---


        # === Draw boxes + arrows with colors ===
        overlay = visualize_gaze_arrows(
            image_pil, 
            bboxes_norm=bboxes_norm,
            gaze_points_xy=gaze_pts_px,
            inout_scores=inout_scores,
            colors=colors
        )
        out_path = os.path.join(args.output_dir, f"{os.path.splitext(img_name)[0]}_gaze.jpg")
        overlay.save(out_path)
        
        frame_elapsed = time.time() - frame_start_time
        print(f"[Done] {img_name} -> {out_path} (Processed in {frame_elapsed:.2f}s)", end='\r')

    # Save the updated posture data back to JSON
    try:
        with open(args.posture_json, 'w') as f:
            json.dump(posture_data, f, indent=2)
        print(f"\n[INFO] Successfully updated posture/gaze data in {args.posture_json}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save updated JSON: {e}")

    # Final Time Calculation
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

    print("\n[INFO] Gaze processing complete.")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")


# 🟥 --- NEW: Main execution block with argparse --- 🟥
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run Gazelle gaze estimation.")
    
    # --- Paths for models ---
    ap.add_argument('--gazelle_checkpoint', type=str, default=DEFAULT_GAZELLE_CHECKPOINT,
                    help='Path to Gazelle checkpoint file.')

    # --- Paths for data ---
    ap.add_argument('--image_folder', type=str, default=DEFAULT_IMAGE_FOLDER,
                    help='Path to the input folder of frames.')
    
    # This argument is required by the script and provided by the bash file
    ap.add_argument('--head_box_json', type=str, default=DEFAULT_HEAD_BOX_JSON,
                    help='Path to the input headbox JSON file.')
    
    ap.add_argument('--posture_json', type=str, default=DEFAULT_POSTURE_JSON_PATH,
                    help='Path to the posture/gesture JSON file (this file will be UPDATED).')
    ap.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                    help='Path to save visualized output images.')
    
    args = ap.parse_args()
    
    main(args)