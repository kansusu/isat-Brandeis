import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0"
import sys
import cv2
import json
import numpy as np
import torch
import time
import math
import argparse
# --- (Setup paths and imports) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "TransT"))
# The following imports are only needed for FULL mode, but they must be present globally
from track_with_transt import setup_detectron2, setup_mmpose 
from mmpose.apis import inference_topdown
from run_pipeline import log_device

from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
from pysot_toolkit.trackers.tracker import Tracker
# --- (End Setup) ---

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE — expected GPU")

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"[GPU] Using {torch.cuda.get_device_name(0)}")


# --- 🟥 GESTURE KEYPOINT INDICES (COCO format) 🟥 ---
L_SHO=5; R_SHO=6; L_ELB=7; R_ELB=8; L_WRI=9; R_WRI=10
L_HIP=11; R_HIP=12
KP_NECK = 1
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
MIN_KP_SCORE = 0.3 

# --- 🟥 GESTURE TUNING PARAMETERS (from gesture_interaction.py) 🟥 ---
GESTURE_CONF_THRESH = 0.25
GESTURE_ELBOW_MIN_DEG = 40.0
GESTURE_ELBOW_MAX_DEG = 70.0
GESTURE_EXT_MIN_RATIO = 0.2
GESTURE_HORIZ_MIN_FRAC = 0.15
GESTURE_HORIZ_MAX_FRAC = 0.6
GESTURE_AWAY_MIN_DOT = -0.8
GESTURE_AWAY_MAX_DOT = 0.1
GESTURE_MIN_ELEV_DEG = -75.0
# --- 🟥 (End Gesture Setup) 🟥 ---


def get_head_box_from_keypoints(keypoints, head_indices=[0, 1, 2, 3, 4]):
    """Helper function to calculate head bounding box from keypoints."""
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    # Add check for keypoints length
    xs = [keypoints[i, 0] for i in head_indices if i < len(keypoints)]
    ys = [keypoints[i, 1] for i in head_indices if i < len(keypoints)]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]

def normalize_box(box, width, height):
    """Helper function to normalize box coordinates to [0, 1]."""
    x1, y1, x2, y2 = box
    return [x1 / width, y1 / height, x2 / width, y2 / height] 

def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def write_placeholder_json(headbox_path: str, posture_path: str) -> None:
    payload = {}
    ensure_parent_dir(headbox_path)
    ensure_parent_dir(posture_path)
    with open(headbox_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(posture_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Saved empty Headbox JSON: {headbox_path}")
    print(f"[INFO] Saved empty Posture/Gesture JSON: {posture_path}")


def classify_posture_angle(neck, left_hip, right_hip):
    """
    Classifies posture based on the angle between neck and mid-hip.
    """
    mid_hip_x = (left_hip[0] + right_hip[0]) / 2
    mid_hip_y = (left_hip[1] + right_hip[1]) / 2
    
    # Calculate vector from neck to hip
    dx = mid_hip_x - neck[0]
    dy = mid_hip_y - neck[1]
    
    angle = np.degrees(np.arctan2(dy, dx))
    
    if dx < 0:
        angle = 180 - angle

    if 65 < angle < 115:
        posture = "neutral"
    elif angle <= 65:
        posture = "lean-in"
    else:
        posture = "lean-out"
        
    return angle, posture

# --- 🟥 GESTURE FUNCTIONS (omitted for brevity, assume unchanged) 🟥 ---
def angle_deg(a, b, c):
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    cosang = np.clip((v1 @ v2) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def length(p, q):
    return float(np.linalg.norm(p - q))

def pointing_class_from_keypoints(
    kpts_xy, kpts_conf,
    person_id_for_debug="P?",
    debug_mode=False,
    elbow_min_deg=GESTURE_ELBOW_MIN_DEG,
    elbow_max_deg=GESTURE_ELBOW_MAX_DEG, 
    ext_min_ratio=GESTURE_EXT_MIN_RATIO,
    conf_thresh=GESTURE_CONF_THRESH,
    horiz_min_frac=GESTURE_HORIZ_MIN_FRAC,
    horiz_max_frac=GESTURE_HORIZ_MAX_FRAC,
    away_min_dot=GESTURE_AWAY_MIN_DOT,
    away_max_dot=GESTURE_AWAY_MAX_DOT,   
    min_elev_deg=GESTURE_MIN_ELEV_DEG
):
    def ok(*idxs):
        return all( (i < kpts_conf.shape[0] and kpts_conf[i] >= conf_thresh) for i in idxs)

    if ok(L_SHO, R_SHO):
        shoulder_width = length(kpts_xy[L_SHO], kpts_xy[R_SHO])
        torso_ctr = (kpts_xy[L_SHO] + kpts_xy[R_SHO]) / 2.0
    else:
        shoulder_width, torso_ctr = 1.0, None
    if shoulder_width < 1.0:
        shoulder_width = 1.0

    def check_arm(shoulder_idx, elbow_idx, wrist_idx, arm_name):
        if not ok(shoulder_idx, elbow_idx, wrist_idx):
            if debug_mode: print(f"[DEBUG] {person_id_for_debug} ({arm_name} arm): FAIL (Keypoints not visible/confident)")
            return None
            
        SHO = kpts_xy[shoulder_idx]; ELB = kpts_xy[elbow_idx]; WRI = kpts_xy[wrist_idx]

        elbow_deg = angle_deg(SHO, ELB, WRI)
        arm_vec = WRI - SHO
        arm_len = np.linalg.norm(arm_vec) + 1e-6
        horiz_frac = abs(arm_vec[0]) / arm_len
        ext_ratio = length(SHO, WRI) / shoulder_width
        elev_deg = math.degrees(math.atan2(-(arm_vec[1]), abs(arm_vec[0]) + 1e-6))

        away_ok = True
        away_dot = 0.0
        if torso_ctr is not None:
            away_dir = SHO - torso_ctr
            away_unit = away_dir / (np.linalg.norm(away_dir) + 1e-6)
            dir_unit  = arm_vec / arm_len
            away_dot  = float(np.clip(dir_unit @ away_unit, -1.0, 1.0))
            away_ok   = (away_dot >= away_min_dot) and (away_dot <= away_max_dot)

        elev_ok  = (elev_deg >= min_elev_deg)
        desk_ok = True 

        is_in_angle_range = (elbow_deg >= elbow_min_deg) and (elbow_deg <= elbow_max_deg)
        extended  = ext_ratio >= ext_min_ratio
        horiz_ok = (horiz_frac >= horiz_min_frac) and (horiz_frac <= horiz_max_frac)

        if debug_mode:
            print(f"[DEBUG] {person_id_for_debug} ({arm_name} arm):")
            print(f"  - elbow_deg: {elbow_deg:.1f} (Range: {elbow_min_deg}-{elbow_max_deg}) -> {'PASS' if is_in_angle_range else 'FAIL'}")
            print(f"  - ext_ratio: {ext_ratio:.2f} (Min: {ext_min_ratio}) -> {'PASS' if extended else 'FAIL'}")
            print(f"  - elev_deg:  {elev_deg:.1f} (Min: {min_elev_deg}) -> {'PASS' if elev_ok else 'FAIL'}")
            print(f"  - away_dot:  {away_dot:.2f} (Range: {away_min_dot} to {away_max_dot}) -> {'PASS' if away_ok else 'FAIL'}")
            print(f"  - horiz_frac:{horiz_frac:.2f} (Range: {horiz_min_frac} to {horiz_max_frac}) -> {'PASS' if horiz_ok else 'FAIL'}")
            print(f"  - desk_ok:   {desk_ok} -> PASS")

        if is_in_angle_range and extended and away_ok and horiz_ok and elev_ok and desk_ok:
            if debug_mode: print(f"  -> RESULT: POINTING")
            return {
                'is_pointing': True, 'arm': arm_name,
                'shoulder': (float(SHO[0]), float(SHO[1])),
                'wrist': (float(WRI[0]), float(WRI[1])),
                'elbow_deg': elbow_deg, 'ext_ratio': ext_ratio,
                'horiz_frac': horiz_frac, 'elev_deg': elev_deg
            }
        
        if debug_mode: print(f"  -> RESULT: NOT POINTING")
        return None

    left  = check_arm(L_SHO, L_ELB, L_WRI, 'left')
    right = check_arm(R_SHO, R_ELB, R_WRI, 'right')
    cand = [c for c in (left, right) if c]
    if not cand:
        return {'is_pointing': False, 'arm': None, 'shoulder': None, 'wrist': None}
    cand.sort(key=lambda c: (c['ext_ratio'], c['horiz_frac'], c['elev_deg']), reverse=True)
    return cand[0]


def draw_pointing_overlay(img, shoulder, wrist, elbow = None, color=(0,255,0)):
    if elbow is None or wrist is None or shoulder is None: 
        return

    p_wrist = (int(wrist[0]), int(wrist[1]))
    cv2.circle(img, p_wrist, 5, color, -1)

    dir_vec = np.array(wrist) - np.array(elbow) 
    
    if np.linalg.norm(dir_vec) < 1e-3:
        return
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
    
    t = 2000.0
    end = (int(p_wrist[0] + t*dir_vec[0]), int(p_wrist[1] + t*dir_vec[1]))
    cv2.arrowedLine(img, p_wrist, end, color, 2, tipLength=0.02)

def find_pointing_target(pointer_person, all_people_data, max_dist_ratio=1.5):
    """
    Checks if a pointing gesture intersects with another person's bounding box.
    """
    pointer_id = pointer_person['pid']
    det = pointer_person['pointing_info']
    kpts = pointer_person['keypoints']
    
    arm_name = det['arm']
    
    elbow_idx = L_ELB if arm_name == 'left' else R_ELB
    wrist_idx = L_WRI if arm_name == 'left' else R_WRI

    if not (kpts.shape[0] > elbow_idx and kpts.shape[0] > wrist_idx):
        return None

    elbow_pt = kpts[elbow_idx][:2]
    wrist_pt = kpts[wrist_idx][:2]
    
    dir_vec = wrist_pt - elbow_pt
    dir_norm = np.linalg.norm(dir_vec)
    if dir_norm < 1e-6:
        return None 
        
    dir_unit = dir_vec / dir_norm
    origin = wrist_pt 

    best_target = None
    min_dist = float('inf')

    for target_person in all_people_data:
        if target_person['pid'] == pointer_id:
            continue 

        target_box = target_person['box']
        target_center = np.array([(target_box[0] + target_box[2]) / 2, 
                                  (target_box[1] + target_box[3]) / 2])
        target_width = target_box[2] - target_box[0]

        vector_to_target = target_center - origin
        
        projection_len = vector_to_target @ dir_unit

        if projection_len < 0:
            continue 

        len_vtt_sq = vector_to_target @ vector_to_target
        perp_dist_sq = len_vtt_sq - projection_len**2
        if perp_dist_sq < 0: perp_dist_sq = 0 
        
        perp_dist = np.sqrt(perp_dist_sq)

        hitbox_radius = target_width * max_dist_ratio

        if perp_dist < hitbox_radius:
            if projection_len < min_dist:
                min_dist = projection_len
                best_target = target_person['pid']

    return best_target
# --- 🟥 (End Gesture Functions) 🟥 ---


def process_image_folder(input_folder, output_headbox_json, output_image_folder, output_posture_json, args):
    """
    Core processing function with mode switch.
    """
    total_start_time = time.time()

    # --- 1. Setup Models ---
    detector = setup_detectron2()
    log_device("Step 2 detectron2")
    
    pose_model, visualizer = None, None
    if args.mode == 'full':
        pose_model, visualizer = setup_mmpose()
        print("[INFO] Running in FULL mode: Posture and Pointing enabled.")
    else:
        print("[INFO] Running in HEADBOX_ONLY mode: Skipping Posture and Pointing analysis.")


    print(f"[INFO] Processing image folder: '{input_folder}'")

    # --- 2. Get and Sort Image Files ---
    try:
        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f"Error: No images found in {input_folder}")
            write_placeholder_json(output_headbox_json, output_posture_json)
            return
    except FileNotFoundError:
        print(f"Error: Input folder not found: {input_folder}")
        write_placeholder_json(output_headbox_json, output_posture_json)
        return
        
    print(f"[INFO] Found {len(image_files)} images to process.")

    # --- 3. Create Output Directory ---
    os.makedirs(output_image_folder, exist_ok=True)
    print(f"[INFO] Saving visualized images to: {output_image_folder}")

    # --- 4. Load First Frame and Initialize Trackers ---
    first_image_path = os.path.join(input_folder, image_files[0])
    first_frame = cv2.imread(first_image_path)
    if first_frame is None:
        print(f"Error: Could not read first image: {first_image_path}")
        write_placeholder_json(output_headbox_json, output_posture_json)
        return

        
    height, width, _ = first_frame.shape

    print("[INFO] Detecting all persons in the first frame...")
    outputs = detector(first_frame)
    # instances = outputs["instances"].to("cpu")
    instances = outputs["instances"]
    
    matched_boxes = instances.pred_boxes.tensor.detach().cpu().numpy()
    
    if matched_boxes is None or len(matched_boxes) == 0:
        write_placeholder_json(output_headbox_json, output_posture_json)
        return


    print(f"[INFO] Found {len(matched_boxes)} persons. Initializing trackers for all.")

    # Initialize trackers
    net_path = os.getenv('TRANST_CHECKPOINT_PATH', '/home/jada4874/multimodal-attention-analyzer/TransT/pytracking/networks/transt.pth.tar')
    trackers = {}
    
    for i, box in enumerate(matched_boxes):
        if box is not None:
            box = list(map(int, box))
            init_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            net = NetWithBackbone(net_path=net_path, use_gpu=True)
            tracker = Tracker(name=f'transt_person_{i}', net=net)
            tracker.initialize(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), {'init_bbox': init_box})
            
            trackers[i] = tracker 

    print(f"[INFO] Created {len(trackers)} trackers.")

    # --- 5. Main Processing Loop (for images) ---
    headbox_results = {}
    posture_results = {} 
    
    # 🟥 Define default labels for headbox_only mode
    SKIPPED_POSTURE_LABEL = "skipped"
    SKIPPED_GESTURE_LABEL = "skipped"

    loop_start_time = time.time()
    total_frames_in_loop = len(image_files)

    for i, image_file in enumerate(image_files):
        frame_idx = i 
        frame_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read image {frame_path}, skipping.")
            continue

        elapsed_loop_time = time.time() - loop_start_time
        avg_fps = (i + 1) / elapsed_loop_time if elapsed_loop_time > 0 else 0
        
        print(f"Processing: {image_file} ({frame_idx+1}/{total_frames_in_loop}) | Avg FPS: {avg_fps:.2f}", end='\r')
        if args.debug:
            print(f"\n--- FRAME: {image_file} ---")
        
        frame_key = image_file
        frame_headbox_results = {}
        frame_posture_results = {}
        people_data_for_frame = []

        # --- 🟥 PASS 1: Track and Detect Headbox (Universal) ---
        for pid, tracker in trackers.items():
            
            # --- Get Bounding Box ---
            if frame_idx == 0:
                box = matched_boxes[pid]
                x1, y1, x2, y2 = map(int, box)
            else:
                out = tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                x, y, w, h = map(int, out['target_bbox'])
                x1, y1, x2, y2 = x, y, x + w, y + h

            current_headbox = None
            head_box_coords = None
            
            # 🟥 Set initial posture/gesture based on mode
            current_posture = SKIPPED_POSTURE_LABEL
            gesture_json_label = SKIPPED_GESTURE_LABEL 
            
            kpts_xy = None
            gesture_det = {'is_pointing': False, 'arm': None, 'shoulder': None, 'wrist': None}


            # --- 🟥 MODE SWITCH POINT ---
            if args.mode == 'full':
                # --- Run MMPose ---
                pose_result = inference_topdown(pose_model, frame, bboxes=[[x1, y1, x2, y2]], bbox_format='xyxy')
                current_posture = "unknown" # Reset to standard full mode default
                
                if pose_result and hasattr(pose_result[0], 'pred_instances') and hasattr(pose_result[0].pred_instances, 'keypoints'):
                    keypoints = pose_result[0].pred_instances.keypoints
                    if isinstance(keypoints, torch.Tensor):
                        keypoints = keypoints.cpu().numpy()

                    has_score = keypoints.shape[2] == 3
                    kpts_xy = keypoints[0, :, :2]
                    kpts_conf = keypoints[0, :, 2] if has_score else np.ones(keypoints.shape[1])

                    # --- Head Box Logic (Requires Keypoints) ---
                    head_box_coords = get_head_box_from_keypoints(keypoints)
                    if head_box_coords:
                        current_headbox = normalize_box(head_box_coords, width, height)
                    
                    # --- Posture Logic (Requires Keypoints) ---
                    if keypoints.shape[1] > max(KP_NECK, KP_LEFT_HIP, KP_RIGHT_HIP):
                        neck_kp = keypoints[0, KP_NECK]
                        left_hip_kp = keypoints[0, KP_LEFT_HIP]
                        right_hip_kp = keypoints[0, KP_RIGHT_HIP]
                        
                        if (not has_score) or \
                           (has_score and neck_kp[2] > MIN_KP_SCORE and 
                            left_hip_kp[2] > MIN_KP_SCORE and 
                            right_hip_kp[2] > MIN_KP_SCORE):
                            
                            _ , current_posture = classify_posture_angle(neck_kp, left_hip_kp, right_hip_kp)

                    # --- Gesture Detection Logic (Requires Keypoints) ---
                    gesture_det = pointing_class_from_keypoints(kpts_xy, kpts_conf, person_id_for_debug=f"P{pid}", debug_mode=args.debug)
                    if gesture_det['is_pointing']:
                        gesture_json_label = f"pointing ({gesture_det['arm']})"
                    else:
                        gesture_json_label = "none"

            else: # HEADBOX_ONLY mode 
                # --- Head Box Logic (Simplified) ---
                x1_h = x1 + (x2 - x1) * 0.2
                y1_h = y1 
                x2_h = x2 - (x2 - x1) * 0.2
                y2_h = y1 + (x2 - x1) * 0.4
                head_box_coords = [x1_h, y1_h, x2_h, y2_h]
                current_headbox = normalize_box(head_box_coords, width, height)
                # Posture/Gesture already set to 'skipped' default
            # --- 🟥 END MODE SWITCH POINT ---


            # Store data for the next pass/saving
            people_data_for_frame.append({
                'pid': pid,
                'box': [x1, y1, x2, y2],
                'keypoints': kpts_xy,
                'pointing_info': gesture_det,
                'head_box_coords': head_box_coords,
                'posture_label': current_posture,
                'headbox_label': current_headbox,
                'gesture_json_label': gesture_json_label
            })

        # --- 🟥 PASS 2: Find Pointing Targets (Only in FULL mode) ---
        all_targets = {}
        if args.mode == 'full':
            for person in people_data_for_frame:
                if person['pointing_info']['is_pointing']:
                    target_id = find_pointing_target(person, people_data_for_frame)
                    all_targets[person['pid']] = target_id

        # --- 🟥 PASS 3: Draw Overlays and Save Dictionaries (Mode-Aware) ---
        for person in people_data_for_frame:
            pid = person['pid']
            head_box = person['head_box_coords']
            posture = person['posture_label']
            gesture_json_label = person['gesture_json_label']

            color = (0,0,255) # Default Red
            label = f"P{pid}"
            
            # Determine drawing coordinates
            if head_box:
                x1h, y1h, x2h, y2h = map(int, head_box)
                cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 0), 2)
                pos_x = x1h
                pos_y_label = y1h - 10
                pos_y_posture = y1h - 35
            else:
                x1,y1,_,_ = person['box']
                pos_x = int(x1)
                pos_y_label = int(y1) - 10
                pos_y_posture = int(y1) - 35
            
            # --- FULL Mode Drawing & Labeling ---
            if args.mode == 'full':
                det = person['pointing_info']
                kpts = person['keypoints']
                
                if det['is_pointing']:
                    target_id = all_targets.get(pid)
                    if target_id is not None:
                        label = f"P{pid} POINTING AT P{target_id}"
                        gesture_json_label = f"pointing at P{target_id}"
                        color = (0, 255, 255) # Yellow
                    else:
                        label = f"P{pid} POINTING ({det['arm']})"
                        color = (0, 255, 0) # Green
                    
                    if kpts is not None:
                        elbow_idx = L_ELB if det['arm'] == 'left' else R_ELB
                        shoulder_idx = L_SHO if det['arm'] == 'left' else R_SHO
                        draw_pointing_overlay(frame, kpts[shoulder_idx], det['wrist'], elbow=kpts[elbow_idx], color=color)
                else:
                    label = f"P{pid} (NOT POINTING)"
                    gesture_json_label = "none"

                # Draw main label (gesture/pointing)
                cv2.putText(frame, label, (pos_x, pos_y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw posture label
                if posture != "unknown":
                    cv2.putText(frame, posture, (pos_x, pos_y_posture), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            
            # --- HEADBOX_ONLY Mode Drawing & Labeling ---
            else:
                label = f"P{pid} (HEADBOX - {SKIPPED_POSTURE_LABEL})" # Use 'skipped' in the display label
                color = (0, 255, 0) # Green for Headbox
                cv2.putText(frame, label, (pos_x, pos_y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # --- Save to Dictionaries ---
            frame_headbox_results[str(pid)] = person['headbox_label']
            
            # 🟥 Save Posture results in all modes (using 'skipped' for headbox_only)
            frame_posture_results[str(pid)] = {
                "posture": posture,
                "gesture": gesture_json_label
            }

        headbox_results[frame_key] = frame_headbox_results
        posture_results[frame_key] = frame_posture_results
        
        name, ext = os.path.splitext(image_file)
        new_filename = f"{name}_posture{ext}"
        
        output_img_path = os.path.join(output_image_folder, new_filename)
        cv2.imwrite(output_img_path, frame)

    print("\n[INFO] Processing complete.")

    # --- 7. Save Final JSONs ---
    with open(output_headbox_json, 'w') as f:
        json.dump(headbox_results, f, indent=2)
        
    # 🟥 Posture JSON is saved regardless, using 'skipped' for headbox_only
    with open(output_posture_json, 'w') as f:
        json.dump(posture_results, f, indent=2)

    print(f"Saved Headbox JSON: {output_headbox_json}")
    print(f"Saved Posture/Gesture JSON: {output_posture_json}")
    print(f"Saved Visualized Images: {output_image_folder}")

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    print(f"\n[INFO] Total execution time: {total_elapsed:.2f} seconds.")


# -------------------------------------------------
# 🟥 Main execution block updated
# -------------------------------------------------
if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_folder', type=str, default="test-1108-10s",
                    help='Path to the input folder of frames')
    ap.add_argument('--output_headbox_json', type=str, default="headboxes_from_test-1108-10s.json",
                    help='Path to save the headbox JSON')
    ap.add_argument('--output_visual_folder', type=str, default="visualized_headboxes",
                    help='Path to save visualized images')
    ap.add_argument('--output_posture_json', type=str, default="posture_test-1108-10s.json",
                    help='Path to save the posture/gesture JSON')
    ap.add_argument('--mode', type=str, choices=['full', 'headbox_only'], default='full',
                    help='Processing mode: "full" (Headbox + Posture + Pointing) or "headbox_only".') 
    ap.add_argument('--debug', action='store_true', help='Print detailed debug values for tuning.')
    
    args = ap.parse_args()
    
    print(f"Running: Processing image folder in '{args.mode.upper()}' mode...")
    
    process_image_folder(
         input_folder=args.input_folder,
         output_headbox_json=args.output_headbox_json,
         output_image_folder=args.output_visual_folder,
         output_posture_json=args.output_posture_json,
         args=args
    )
    print("\n=== Processing Complete. Check JSON and image folder. ===")