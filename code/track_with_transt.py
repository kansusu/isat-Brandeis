import math
import os
import cv2
import numpy as np
import csv
import pickle
import sys
import scipy.signal
from collections import defaultdict
from run_pipeline import log_device
import traceback
import torch


# === TransT Setup ===
sys.path.append('./TransT')
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
from pysot_toolkit.trackers.tracker import Tracker


from collections import defaultdict, deque

hand_motion_history = defaultdict(lambda: deque(maxlen=15))
wrist_history = defaultdict(lambda: deque(maxlen=15))
# === MediaPipe for smile + speaking ===
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)
from scipy.signal import savgol_filter

_orig_torch_load = torch.load
def _torch_load_no_weights_only(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_no_weights_only

def detect_shakes_per_person(head_x_by_frame, fps):
    shake_events = defaultdict(list)
    for pid, records in head_x_by_frame.items():
        records.sort(key=lambda x: x[0])
        frames = [r[0] for r in records]
        x_vals = [r[3] for r in records]
        #print('x_vals:',x_vals)
        if len(x_vals) < 5:
            continue
        x_smooth = savgol_filter(x_vals, 7, 3)
        #print('x_smooth:',x_smooth)


        amp_thresh = 20
        prev_extrema = None

        extrema_frames = []

        for i in range(1, len(x_smooth) - 1):
            prev = x_smooth[i - 1]
            curr = x_smooth[i]
            next_ = x_smooth[i + 1]
            frame = frames[i]

            is_peak = curr > prev and curr > next_
            is_valley = curr < prev and curr < next_

            if is_peak or is_valley:
                if prev_extrema is not None:
                    delta = abs(curr - prev_extrema[1])
                    #print('delta',delta)
                    if delta > amp_thresh:
                        extrema_frames.append(frame)
                prev_extrema = (frame, curr)
        shake_events[pid] = extrema_frames
            #elif prev > curr < next_ and (prev - curr) > 0.3 and (next_ - curr) > 0.3:
               # shake_events[person_id].append(frames[i])
    return shake_events
        


def detect_smile(keypoints, mouth_thresh=1.5, openness_thresh=0.4):
     
    
    #if keypoints.shape[0] <= max(LEFT_MOUTH, RIGHT_MOUTH, UPPER_LIP, LOWER_LIP):
    #    return False
    
    left = keypoints[0,76]
    right =keypoints[0,82]
    upper = keypoints[0,79]
    lower = keypoints[0,85]

    mouth_width = np.linalg.norm(right - left)
    mouth_open = np.linalg.norm(lower - upper)
    smile_ratio = mouth_width / (mouth_open + 1e-5)
    #print('smile_ratio',smile_ratio)
    #print('mouth_thresh / openness_thresh',mouth_thresh / openness_thresh)
    return smile_ratio > 2.5


# === Detectron2 + MMPose Setup ===
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from mmpose.apis import inference_topdown, init_model, init_model

# from mmpose.apis import inference_topdown, init_model
# from mmpose.apis import vis_pose_result
from mmengine.registry import DefaultScope
# from mmdet.apis import init_detector, inference_detector

def setup_detectron2():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

COCO_CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]

# def setup_mmdet():
#     with DefaultScope.overwrite_default_scope('mmdet'):
#         config = 'mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
#         checkpoint = 'rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
#         model = init_detector(config, checkpoint, device='cuda:0')
#     return model

def setup_mmdet():
    from mmengine.registry import DefaultScope
    from mmdet.apis import init_detector

    with DefaultScope.overwrite_default_scope('mmdet'):
        config = 'mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
        checkpoint = 'rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
        model = init_detector(config, checkpoint, device='cuda:0')

    return model


def setup_mmpose():
    from mmengine.registry import DefaultScope
    from mmpose.apis import init_model

    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.join(
        repo_root,
        "mmpose",
        "configs",
        "body_2d_keypoint",
        "rtmpose",
        "coco",
        "rtmpose-m_8xb256-420e_coco-256x192.py",
    )
    default_ckpt = os.path.join(
        repo_root,
        "weights",
        "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth",
    )

    config = os.getenv("MMPOSE_CONFIG", default_cfg)
    checkpoint = os.getenv("MMPOSE_CHECKPOINT", default_ckpt)

    if not os.path.isfile(config):
        raise FileNotFoundError(
            f"MMPOSE_CONFIG not found: {config}. "
            "Set MMPOSE_CONFIG to a valid config path."
        )
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"MMPOSE_CHECKPOINT not found: {checkpoint}. "
            "Download the checkpoint or set MMPOSE_CHECKPOINT."
        )

    with DefaultScope.overwrite_default_scope("mmpose"):
        model = init_model(config, checkpoint, device="cuda:0")

    return model, None

# def setup_mmpose_hand():
#     with DefaultScope.overwrite_default_scope('mmpose'):
#         config = 'mmpose/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py'
#         checkpoint = 'rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
#         model = init_model(config, checkpoint, device='cuda:0')
#     return model

def setup_mmpose_hand():
    from mmengine.registry import DefaultScope
    from mmpose.apis import init_model

    with DefaultScope.overwrite_default_scope('mmpose'):
        config = 'mmpose/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py'
        checkpoint = 'rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
        model = init_model(config, checkpoint, device='cuda:0')

    return model

#typing
def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2
#gesture
def get_hand_boxes_from_pose(keypoints, img_w, img_h, box_size=80):
    hand_boxes = []
    if keypoints is None or len(keypoints) == 0:
        return hand_boxes

    for i in [9, 10]:

        if len(keypoints[0, i]) < 2:
            continue
        try:
            if len(keypoints[0, i]) == 2:
                x, y = keypoints[0, i]
                score = 1.0
            else:
                x, y, score = keypoints[0, i]
    
            if (score < 0.3 or np.isnan(x) or np.isnan(y) or x < 0 or y < 0 or x > img_w or y > img_h):
                continue
            x1 = max(int(x - box_size / 2), 0)
            y1 = max(int(y - box_size / 2), 0)
            x2 = min(int(x + box_size / 2), img_w)
            y2 = min(int(y + box_size / 2), img_h)
            if x2 > x1 and y2 > y1 and (x2-x1) > 10 and (y2-y1) > 10:
                hand_boxes.append([x1, y1, x2, y2])
        except:
            continue
    return hand_boxes
#palm

def detect_palm_up(hand_kpts):
    if hand_kpts.ndim == 3:
        hand_kpts = hand_kpts[0]
    if hand_kpts.shape[0] < 21:
        return False
    wrist = hand_kpts[0]
    #tips = [hand_kpts[i] for i in [4, 8, 12, 16, 20]]
    tips = [hand_kpts[ i] for i in [4, 8, 12, 16, 20]]
    above_wrist = all(tip[1] < wrist[1] for tip in tips)
    finger_spread = abs(hand_kpts[8][0] - hand_kpts[20][0]) > 30

    return above_wrist and finger_spread

#wavng
#from collections import deque
#hand_motion_history = defaultdict(lambda: deque(maxlen=15))  # per person/hand

def detect_waving(pid, frame_idx, hand_kpts, history, sway_thresh=30, min_oscillations=1.5):
    if hand_kpts.ndim == 3:
        hand_kpts = hand_kpts[0] 
    if hand_kpts.shape[0] < 13:
        return False

    wrist = hand_kpts[0]
    wrist_x, wrist_y = float(wrist[0]), float(wrist[1])

    if pid not in history:
        history[pid] = []
    history[pid].append(wrist_x)


    if len(history[pid]) > 15:
        history[pid] = history[pid][-15:]

    xs = history[pid]

    sway_range = max(xs) - min(xs)
    if sway_range < sway_thresh:
        return False

    #left_shoulder = float(hand_kpts[11][1])
    #right_shoulder = float(hand_kpts[12][1])
    #shoulder_y = min(left_shoulder, right_shoulder)
    #shoulder = hand_kpts[12] if hand_kpts[12][1] < hand_kpts[11][1] else hand_kpts[11]
    #if wrist_y > float(shoulder_y) + 20:
    #    return False

   # hand_motion_history[person_id].append((frame_idx, wrist_x))
   # if len(hand_motion_history[person_id]) < 20:
    #    return False

    #xs = [x for _, x in hand_motion_history[person_id]]
    #delta_x = np.diff(xs)

    #direction_changes = sum((a * b < 0) for a, b in zip(delta_x, delta_x[1:]))
    #swing_range = max(xs) - min(xs)
    diffs = np.diff(xs)
    signs = np.sign(diffs)
    zero_crossings = np.sum(np.diff(signs) != 0)

    return zero_crossings >= min_oscillations * 2

#Slicing Motion
def detect_slicing(pid, frame_idx, hand_kpts, history, axis_thresh=20, y_dom_ratio=1.3):
    if hand_kpts is None or len(hand_kpts) < 1:
        return False
    if isinstance(hand_kpts, np.ndarray):
        if hand_kpts.ndim == 3:
            hand_kpts = hand_kpts[0]

    if hand_kpts.shape[0] < 6:
        return False

    left_wrist = hand_kpts[0]
    right_wrist = hand_kpts[1]
    lw_x, lw_y = left_wrist[0], left_wrist[1]
    rw_x, rw_y = right_wrist[0], right_wrist[1]

    if pid not in history:
        history[pid] = {"lw_x": [], "lw_y": [], "rw_x": [], "rw_y": []}
    
    history[pid]["lw_x"].append(lw_x)
    history[pid]["lw_y"].append(lw_y)
    history[pid]["rw_x"].append(rw_x)
    history[pid]["rw_y"].append(rw_y)

    max_len = 7
    for k in history[pid]:
        history[pid][k] = history[pid][k][-max_len:]

    if len(history[pid]["lw_x"]) < 5:
        return False
    
    lw_x_range = max(history[pid]["lw_x"]) - min(history[pid]["lw_x"])
    rw_x_range = max(history[pid]["rw_x"]) - min(history[pid]["rw_x"])
    lw_y_range = max(history[pid]["lw_y"]) - min(history[pid]["lw_y"])
    rw_y_range = max(history[pid]["rw_y"]) - min(history[pid]["rw_y"])

    left_vertical = lw_y_range > axis_thresh and lw_y_range > lw_x_range * y_dom_ratio
    right_vertical = rw_y_range > axis_thresh and rw_y_range > rw_x_range * y_dom_ratio

    return left_vertical and right_vertical
    

    #left_y_movement = max(h["left_y"]) - min(h["left_y"])
    #right_y_movement = max(h["right_y"]) - min(h["right_y"])
    #left_x_variation = max(h["left_x"]) - min(h["left_x"])
    #right_x_variation = max(h["right_x"]) - min(h["right_x"])

    #vertical = left_y_movement > y_thresh and right_y_movement > y_thresh
    #horizontal_stable = left_x_variation < x_thresh and right_x_variation < x_thresh
    #return verti#cal and horizontal_stable




#pointing



def is_arm_raised(wrist_y, shoulder_y, min_diff=20):
    return wrist_y < (shoulder_y - min_diff)

def detect_pointing(pid, hand_kpts,extended_thresh=30, curled_thresh=25):
    if hand_kpts is None or len(hand_kpts) < 1:
        return False
    if isinstance(hand_kpts, np.ndarray):
        if hand_kpts.ndim == 3:
            hand_kpts = hand_kpts[0]
        if hand_kpts.shape[0] < 21:
            return False
    
   # tip_ids = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
   # base_ids = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    
   # middle_len = np.linalg.norm(hand_kpts[tip_ids["middle"]][:2] - hand_kpts[base_ids["middle"]][:2])
   # index_len = np.linalg.norm(hand_kpts[tip_ids["index"]][:2] - hand_kpts[base_ids["index"]][:2])
   # ring_len = np.linalg.norm(hand_kpts[tip_ids["ring"]][:2] - hand_kpts[base_ids["ring"]][:2])
   # pinky_len = np.linalg.norm(hand_kpts[tip_ids["pinky"]][:2] - hand_kpts[base_ids["pinky"]][:2])
    wrist = hand_kpts[0][:2]
    index = hand_kpts[8][:2]
    middle = hand_kpts[12][:2]
    ring   = hand_kpts[16][:2]
    pinky  = hand_kpts[20][:2]

    index_len = np.linalg.norm(index - wrist)

    middle_len = np.linalg.norm(middle - wrist)

    ring_len = np.linalg.norm(ring - wrist)
    pinky_len = np.linalg.norm(pinky - wrist)
    log_device("Step 2 tracker init")
    curled_count = sum([middle_len < curled_thresh,ring_len < curled_thresh,pinky_len < curled_thresh])
    pointing_like = (index_len > extended_thresh and curled_count >= 2 and index_len > middle_len and
            index_len > ring_len and index_len > pinky_len)
    return index_len > extended_thresh and middle_len < curled_thresh




#raise
from scipy.signal import find_peaks
wrist_history = defaultdict(lambda: deque(maxlen=15))
def detect_raise_sway(person_id, frame_idx, hand_kpts,rise_thresh=20, min_sway_range=60,min_peak_prominence=10, min_peaks=2):
    if hand_kpts is None or len(hand_kpts) < 1:
        return False
    if isinstance(hand_kpts, np.ndarray):
        if hand_kpts.ndim == 3:
            hand_kpts = hand_kpts[0]
        if hand_kpts.shape[0] < 13:
            return False
    
    wrist_x = float(hand_kpts[0][0])
    wrist_y = hand_kpts[0][1]
    left_shoulder_y = float(hand_kpts[11][1])
    right_shoulder_y = float(hand_kpts[12][1])
    shoulder_y = min(left_shoulder_y, right_shoulder_y)

    if wrist_y < shoulder_y:
        return False
    wrist_history[person_id].append((frame_idx, (wrist_x, wrist_y)))

    if len(wrist_history[person_id]) < 10:
        return False
    xs = [pt[0] for _, pt in wrist_history[person_id]]
    ys = [pt[1] for _, pt in wrist_history[person_id]]

    vertical_change = ys[0] - ys[-1]
    if vertical_change < rise_thresh:
        return False
    
    horizontal_range = max(xs) - min(xs)
    if horizontal_range < min_sway_range:
        return False
    #duration = frames[-1] - frames[0] + 1
    
    xs_np = np.array(xs)
    peaks, peak_props = find_peaks(xs_np, prominence=min_peak_prominence, distance=2)
    valleys, valley_props = find_peaks(-xs_np, prominence=min_peak_prominence, distance=2)

    #total_peaks = len(peaks) + len(valleys)
    #horizontal_range = max(xs) - min(xs)

    # Horizontal sway check
    #xs_np = np.array(xs)
    #peaks, _ = find_peaks(xs_np, distance=2)
    #valleys, _ = find_peaks(-xs_np, distance=2)
    #if len(peaks) + len(valleys) < min_peaks:
    #    return False
    #sway_range = max(xs) - min(xs)
    #if total_peaks < min_peaks or horizontal_range <  min_sway_range:
    #    return False
    if len(peaks) >= 3 and len(valleys) >= 3:
        return True
    return False


    


from scipy.signal import savgol_filter, find_peaks

def detect_nods_per_person(head_y_by_frame, fps):
    
    nod_events = defaultdict(list)

    for person_id, records in head_y_by_frame.items():
        records.sort(key=lambda x: x[0])
        frames = [r[0] for r in records]
        y_vals = [r[2] for r in records]

        if len(y_vals) <5:
            continue
    
        y_smooth = savgol_filter(y_vals, 7, 3)

        for i in range(len(y_smooth) - 2):
            prev = y_smooth[i - 1]
            curr = y_smooth[i]
            next_ = y_smooth[i + 1]
    
            if prev > curr < next_ and (prev - curr) > 0.3 and (next_ - curr) > 0.3:
                nod_events[person_id].append(frames[i])

    return nod_events





def classify_posture_angle(neck, left_hip, right_hip):
    mid_hip_x = (left_hip[0] + right_hip[0]) / 2
    mid_hip_y = (left_hip[1] + right_hip[1]) / 2
    dx = mid_hip_x - neck[0]
    dy = mid_hip_y - neck[1]
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 70:
        return angle, "lean-in"
    elif angle > 90:
        return angle, "lean-out"
    return angle, "neutral"


#pointing 
def detect_pointing_test(pid, hand_kpts,finger_thresh=25):
    if hand_kpts is None or len(hand_kpts) < 1:
        return False
    if isinstance(hand_kpts, np.ndarray):
        
        if hand_kpts.ndim == 3:
            hand_kpts = hand_kpts[0]
        if hand_kpts.shape[0] < 21:
            return False

    wrist = hand_kpts[0]
    index_tip = hand_kpts[8]
    middle_tip = hand_kpts[12]
    ring_tip = hand_kpts[16]
    pinky_tip = hand_kpts[20]

    dist_index = np.linalg.norm(index_tip - wrist)
    dist_others = np.mean([
        np.linalg.norm(middle_tip - wrist),
        np.linalg.norm(ring_tip - wrist),
        np.linalg.norm(pinky_tip - wrist)
    ])
    return (dist_index - dist_others) > finger_thresh
    #mcp = {"index": 5, "middle": 9,"ring": 13,"pinky": 17}
    #tip = {"index": 8, "middle": 12,"ring": 16,"pinky": 20}
    #def is_extended(finger):
     #   base = hand_kpts[mcp[finger]]
     #   tip_pt = hand_kpts[tip[finger]]
     #   return np.linalg.norm(tip_pt[:2] - base[:2]) > (threshold_tip if finger == "index" else threshold_fold)

    #index_extended = is_extended("index")
    #middle_folded = not is_extended("middle")
    #ring_folded = not is_extended("ring")
    #pinky_folded = not is_extended("pinky")

    #return index_extended and middle_folded and ring_folded and pinky_folded

def draw_pointing_vector(image, wrist, index_tip, color=(0, 0, 255), length=200):
    x1, y1 = wrist
    x2, y2 = index_tip

    direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return image

    direction = direction / norm
    x3 = int(x1 + direction[0] * length)
    y3 = int(y1 + direction[1] * length)

    return cv2.arrowedLine(image, (int(x1), int(y1)), (x3, y3), color, 2, tipLength=0.2)
    
def enlarged_box(box, scale=1.3):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

def boxes_overlap(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


object_tracker = {} 
object_bbox_tracker = {}
object_angle_tracker = {}
object_movement_log = []
prev_object_positions = {}

distance_threshold = 25
angle_change_threshold = 20 
iou_threshold = 0.7 

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA+1) * max(0, yB - yA+1)
    if interArea ==0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]+1) * (boxA[3] - boxA[1]+1)
    boxBArea = (boxB[2] - boxB[0]+1) * (boxB[3] - boxB[1]+1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def compute_angle_change(boxA, boxB):
    def angle(w, h):
        return math.degrees(math.atan2(h, w))
    wA, hA = boxA[2] - boxA[0], boxA[3] - boxA[1]
    wB, hB = boxB[2] - boxB[0], boxB[3] - boxB[1]

    return abs(angle(wA, hA) - angle(wB, hB))


def match_detections_to_trackers(detections, trackers, iou_thresh=0.3):
    matched = {}
    unmatched_dets = []
    unmatched_trks = set(trackers.keys())

    for det in detections:
        best_iou = 0
        best_pid = None
        for pid in trackers:
            if hasattr(trackers[pid], 'latest_bbox'):
                pred_box = trackers[pid].latest_bbox
                i = compute_iou(det, pred_box)
                if i > best_iou:
                    best_iou = i
                    best_pid = pid
        if best_iou > iou_thresh:
            matched[best_pid] = det
            unmatched_trks.discard(best_pid)
        else:
            unmatched_dets.append(det)
    return matched, unmatched_dets, list(unmatched_trks)

def detect_persons(frame, predictor):
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    person_boxes = [list(map(int, box)) for box, cls in zip(boxes, classes) if cls == 0]  # Class 0 = person
    return person_boxes


def detect_waving_and_slicing(pid, frame_idx, hand_kpts, wave_history, slice_history,min_motion_frames=5, x_thresh_wave=40, y_thresh_wave=200,
        y_thresh_slice=18, x_thresh_slice=300):
    if hand_kpts is None or len(hand_kpts)< 1:
        return False,False

    if isinstance(hand_kpts, np.ndarray) and hand_kpts.ndim == 3:
        hand_kpts = hand_kpts[0]
   # if len(hand_kpts) < 1:
   #     return False, False

    lwrist_x, lwrist_y = hand_kpts[0][0], hand_kpts[0][1]
    rwrist_x, rwrist_y = hand_kpts[1][0], hand_kpts[1][1]

    if pid not in wave_history:
        #wave_history[pid] ={'lwrist': [], 'rwrist': []}
        wave_history[pid] = {'lx': [], 'ly': [], 'rx': [], 'ry': []}
    if pid not in slice_history:
        #slice_history[pid] ={'lwrist': [], 'rwrist': []}
        slice_history[pid] = {'lx': [], 'ly': [], 'rx': [], 'ry': []}
    
    wh = wave_history[pid]
    sh = slice_history[pid]

    wh['lx'].append(lwrist_x)
    wh['ly'].append(lwrist_y)
    wh['rx'].append(rwrist_x)
    wh['ry'].append(rwrist_y)

    sh['lx'].append(lwrist_x)
    sh['ly'].append(lwrist_y)
    sh['rx'].append(rwrist_x)
    sh['ry'].append(rwrist_y)

    for key in wh:
        wh[key] = wh[key][-min_motion_frames:]
    for key in sh:
        sh[key] = sh[key][-min_motion_frames:]
    
    waving,slicing = False, False
    #wave_history[pid].append(((lwrist_x + rwrist_x) / 2, (lwrist_y + rwrist_y) / 2))
    #slice_history[pid].append(((lwrist_x + rwrist_x) / 2, (lwrist_y + rwrist_y) / 2))
    def get_motion(track, axis):
        vals = [p[axis] for p in track]
        return max(vals) - min(vals) if len(vals) > 1 else 0

    for side in ['l', 'r']:
        x_range = max(wh[f"{side}x"]) - min(wh[f"{side}x"]) if len(wh[f"{side}x"]) >= min_motion_frames else 0
        y_range = max(wh[f"{side}y"]) - min(wh[f"{side}y"]) if len(wh[f"{side}y"]) >= min_motion_frames else 0
       # print("x_range, y_range:",x_range, y_range)
        if 150 <x_range <300 and 160< y_range < 200:
            waving = True
            break
    for side in ['l', 'r']:
        x_range = max(sh[f"{side}x"]) - min(sh[f"{side}x"]) if len(sh[f"{side}x"]) >= min_motion_frames else 0
        y_range = max(sh[f"{side}y"]) - min(sh[f"{side}y"]) if len(sh[f"{side}y"]) >= min_motion_frames else 0
        #print("x_range, y_range:",x_range, y_range)
        if 18 <y_range < 125 and 180<x_range< 274:
            slicing = True
            break


    return waving, slicing




def process_video_with_transt(input_path, output_csv):
    detector = setup_detectron2()
    pose_model, visualizer = setup_mmpose()
    det_model = setup_mmdet()
    hand_model = setup_mmpose_hand()
    net_path = '/home/ubuntu/multimodal-attention-analyzer/TransT/pytracking/networks/transt.pth'
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")


    output_path = "person_bboxes_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1:
        print("FPS fallback: setting to 30 manually.")
        fps=30
    else:
        print(f"Video FPS detected: {fps}")
    ret, first_frame = cap.read()
    assert ret

    outputs = detector(first_frame)
    instances = outputs["instances"].to("cpu")
    pred_boxes = instances.pred_boxes.tensor.numpy()
    #if len(pred_boxes) > 3:
    #   pred_boxes = pred_boxes[:3]   
    frame_idx = 0
    next_pid = 0
    trackers = {}
    #for i, box in enumerate(pred_boxes):
    #    box = list(map(int, box))
    #    init_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    #    net = NetWithBackbone(net_path=net_path, use_gpu=True)
    #    tracker = Tracker(name=f'transt_{i}', net=net)
    #    tracker.initialize(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), {'init_bbox': init_box})
    #    trackers[i] = tracker

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)


    waving_history = {}
    slicing_history= {}
    person_bboxes = defaultdict(list)
    head_y_by_person = defaultdict(list)
    full_log = []
    object_log = []
    head_shaking_history={}
    pointing_target = ""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        draw_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = frame_idx / fps
        
        detections = detect_persons(frame, detector)
        matched, unmatched_dets, _ = match_detections_to_trackers(detections, trackers)

        
        iou_thresh =0.7
        with DefaultScope.overwrite_default_scope('mmdet'):
            from mmdet.apis import inference_detector
            object_results = inference_detector(det_model, frame)
            laptop_boxes = []
            if hasattr(object_results, 'pred_instances'):
                pred_instances = object_results.pred_instances
                for i in range(len(pred_instances.bboxes)):
                    label = int(pred_instances.labels[i])
                    if label == 63:  # laptop
                        bbox = pred_instances.bboxes[i].cpu().numpy()
                        score = float(pred_instances.scores[i])
                        if score > 0.5:
                            laptop_boxes.append(bbox)

            if hasattr(object_results, 'pred_instances'):
                pred_instances = object_results.pred_instances
        #else:
           # pred_instances = object_results 
        # for class_id, detections in enumerate(object_results):
                for i in range(len(pred_instances.bboxes)):
                    bbox = pred_instances.bboxes[i].cpu().numpy()
                    score = float(pred_instances.scores[i])
                    label = int(pred_instances.labels[i])
                    x1, y1, x2, y2 = bbox
            
                    if score > 0.5:
                        #object_log.append([frame_idx, f"{timestamp:.2f}", label, x1, y1, x2, y2, score])
                        label_name = COCO_CLASSES[label] if 0 <= label < len(COCO_CLASSES) else f"class_{label}"
                        object_log.append([frame_idx, f"{timestamp:.2f}", label, label_name, x1, y1, x2, y2, score])
                        track_key = (label_name, i)

                        if track_key in prev_object_positions:
                            prev_bbox = prev_object_positions[track_key]
                            prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
                            curr_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                            dist = np.linalg.norm(np.array(prev_center) - np.array(curr_center))

                            prev_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
                            curr_area = (x2 - x1) * (y2 - y1)
                            area_ratio = min(prev_area, curr_area) / max(prev_area, curr_area)

                            iou = compute_iou(prev_bbox, [x1, y1, x2, y2])
                            is_furniture = any(name in label_name.lower() for name in ["chair", "couch", "dining table"])

                            dist_thresh = 150 if is_furniture else 1
                            angle_thresh = 1
                            angle_change = compute_angle_change(prev_bbox, [x1, y1, x2, y2])
                            if dist > dist_thresh and area_ratio < 0.9 and iou!= 0.00 and angle_change > angle_thresh:
                                #print(f"[Frame {frame_idx}] {label_name}_{i} moved (Δ={dist:.1f}, IoU={iou:.2f}, AreaRatio={area_ratio:.2f})")
                                object_movement_log.append([frame_idx, f"{timestamp:.2f}", label_name, i, dist, iou, area_ratio])
                        prev_object_positions[track_key] = [x1, y1, x2, y2]        
             
        for pid, tracker in matched.items():
            try:
                print('we have tracker')
                tracker = trackers[pid]
                tracker_output = tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print('tracker for matched ones')
        
                x, y, w, h = map(int, tracker_output["target_bbox"])
                x2, y2 = x + w, y + h
                tracker.latest_bbox = [x, y, x2, y2]
            
                cv2.rectangle(draw_frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(draw_frame, f"PID {pid}", (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                full_log.append([frame_idx, pid, x, y, x2, y2])
            except Exception as e:
                print(f"[Frame {frame_idx}] Tracker error for ID {pid}: {type(e).__name__}: {e}")
            for det in unmatched_dets:
                x1, y1, x2, y2 = det
                init_box = [x1, y1, x2 - x1, y2 - y1]
                net = NetWithBackbone(net_path=net_path, use_gpu=True)
                tracker = Tracker(name=f'transt_{next_pid}', net=net)
                tracker.initialize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), {'init_bbox': init_box})
                tracker.latest_bbox = [x1, y1, x2, y2]
                trackers[next_pid] = tracker

                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(draw_frame, f"PID {next_pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                full_log.append([frame_idx, next_pid, x1, y1, x2, y2])
                next_pid += 1
            video_writer.write(draw_frame)
        


               # smile, speaking = detect_smile_and_speaking(frame, (x, y, x2, y2))
               # smile = detect_smile(keypoints, mouth_thresh=1.5, openness_thresh=0.4)
            with DefaultScope.overwrite_default_scope('mmpose'):
                    pose_result = inference_topdown(
                        pose_model,
                        frame,
                        bboxes=[[x, y, x2, y2]],
                        bbox_format='xyxy'
                    )
                
                #if pose_result and isinstance(pose_result, list) and 'keypoints' in pose_result[0] and len(pose_result[0]['keypoints']) > 0:
            hand_boxes = []
            if pose_result and hasattr(pose_result[0], 'pred_instances') and hasattr(pose_result[0].pred_instances, 'keypoints'):

                keypoints = pose_result[0].pred_instances.keypoints
                if hasattr(keypoints, 'cpu'):
                    keypoints = keypoints.cpu().numpy()
                    img_h, img_w = frame.shape[:2]
                    hand_boxes = get_hand_boxes_from_pose(keypoints, img_w, img_h)
                else:
                    continue
                enlarged_laptop_boxes = [enlarged_box(box, scale=1.2) for box in laptop_boxes]
                hand_on_laptop = any(
                        boxes_overlap(hand_box, laptop_box)
                        for hand_box in hand_boxes
                        for laptop_box in enlarged_laptop_boxes
                        )
                #print(f"[Frame {frame_idx}] PID {pid} keypoints shape: {keypoints.shape}") 
                #print('keypoints:', keypoints)
                #print('start smile detection...')
                smile = detect_smile(keypoints, mouth_thresh=2.5, openness_thresh=0.4)
                #print('end smile ')
                head_x = float(keypoints[0, 0, 0])
                head_y = float(keypoints[0,0,1])
                #shaking = detect_head_shaking(pid, frame_idx, keypoints, head_shaking_history)
                #print(f"Frame {frame_idx} PID {pid} - Wrist positions: Left {keypoints[0,9]}, Right {keypoints[0,10]}")

                #print(f"Generated hand boxes: {hand_boxes}")
                gesture_labels = {"palm_up": 0, "wave": 0, "slice": 0, "raise_sway": 0, "pointing": 0}
                for hand_box in hand_boxes:
                    if (hand_box is None or len(hand_box) != 4 or any(v is None or np.isnan(v) for v in hand_box) or hand_box[2] <= hand_box[0] or hand_box[3] <= hand_box[1]):
                        continue
                    try:
                        #hand_box = [float(v) for v in hand_box]
                        x1, y1, x2, y2 = [float(v) for v in hand_box]
                        if x1 >= x2 or y1 >= y2:
                            continue
                    
                    
                        #bbox_dict = {'bbox': np.array([x1, y1, x2, y2],dtype=np.float32), 'bbox_score': np.array([1.0], dtype=np.float32)}
                        try:
                            hand_result = inference_topdown(hand_model, frame, [[x1, y1, x2, y2]], bbox_format='xyxy')
                            if not hand_result or len(hand_result) ==0:
                                continue
                            if not hasattr(hand_result[0], 'pred_instances'):
                                continue

                            hand_kpts = hand_result[0].pred_instances.keypoints
                            waving, slicing = detect_waving_and_slicing(pid, frame_idx, hand_kpts, waving_history, slicing_history)
                            if hand_kpts is None or len(hand_kpts) == 0:
                                continue
                            if hasattr(hand_kpts, 'cpu'):
                                hand_kpts = hand_kpts.cpu().numpy()
                            if hand_kpts.size == 0 or np.isnan(hand_kpts).any():
                                continue
                            if detect_palm_up(hand_kpts):
                                gesture_labels["palm_up"] = 1
                            if detect_pointing(pid, hand_kpts,extended_thresh=25, curled_thresh=45):
                                gesture_labels["pointing"] = 1
                                pointing_target = ""
                                if gesture_labels["pointing"]:
                                    wrist = hand_kpts[0][0][:2]  # wrist
                                    index_tip = hand_kpts[0][8][:2]  # index finger tip
                                    direction = index_tip - wrist
                                    direction_norm = np.linalg.norm(direction)
                                    if direction_norm > 80:
                                        direction = direction / direction_norm

                                        min_angle = float('inf')
                                        best_object = None

                                        for obj in object_log:
                                            obj_frame, _, _, label_name, x1, y1, x2, y2, _ = obj
                                            if int(obj_frame) != frame_idx:
                                                continue
                                            obj_center = np.array([(float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2])
                                            vec_to_obj = obj_center - wrist
                                            vec_to_obj_norm = np.linalg.norm(vec_to_obj)
                                            #print('vec_to_obj_norm:',vec_to_obj_norm)
                                            if vec_to_obj_norm < 30 or vec_to_obj_norm >700:
                                                continue
                                            vec_to_obj = vec_to_obj / vec_to_obj_norm

                                            angle = np.arccos(np.clip(np.dot(direction, vec_to_obj), -1.0, 1.0))
                                            angle_deg = np.degrees(angle)
                                            
                                            proj_len = np.dot(vec_to_obj, direction)
                                            proj_point = wrist + proj_len * direction
                                            perp_dist = np.linalg.norm(obj_center - proj_point)
                                            print(f"[Frame {frame_idx}] PID {pid} → {label_name}: angle={angle_deg:.1f}, perp_dist={perp_dist:.1f}, vec_to_obj={np.round(vec_to_obj, 1)}")

                                            #print(f"[Frame {frame_idx}] PID {pid} → {label_name}: angle={angle_deg:.1f}, perp_dist={perp_dist:.1f}, vec_to_obj={vec_to_obj:.1f}")
    
                                            if angle_deg < 12 and perp_dist <80:
                                                score = angle_deg + 0.2 * perp_dist + 0.01 * obj_dist
                                                if score < best_score:
                                                    best_score = score
                                                    best_object = label_name

                                        if best_object:
                                            pointing_target = best_object
                                            print(f"[Frame {frame_idx}] PID {pid} is pointing at: {pointing_target}")

                            #if detect_waving(pid, frame_idx, hand_kpts,waving_history):
                            if waving:    
                                gesture_labels["wave"] = 1
                            if slicing:
                            #if detect_slicing(pid, frame_idx, hand_kpts,slicing_history):
                                gesture_labels["slice"] = 1
                            if detect_raise_sway(pid, frame_idx, hand_kpts,rise_thresh=20, min_sway_range=90,min_peak_prominence=15, min_peaks=2):
                                gesture_labels["raise_sway"] = 1
                        except Exception as e:
                            import traceback
                            print(f"Hand detection error for PID {pid} at frame {frame_idx}: {type(e).__name__}: {e}")
                            traceback.print_exc()
                            continue
                    except Exception as e:
                        print(f"Hand box processing error for PID {pid} at frame {frame_idx}: {str(e)}")
                        continue
                neck = neck = [float(keypoints[0, 1, 0]), float(keypoints[0, 1, 1])]
                left_hip = [float(keypoints[0, 11, 0]), float(keypoints[0, 11, 1])]
                right_hip = [float(keypoints[0, 12, 0]), float(keypoints[0, 12, 1])]
                angle, posture = classify_posture_angle(neck, left_hip, right_hip)
                head_y_by_person[pid].append((frame_idx, timestamp, head_y,head_x))
            
                full_log.append([
                        frame_idx, f"{timestamp:.2f}", pid,# f"{head_y:.2f}",
                        #f"{angle:.2f}",
                        posture, int(smile),int(hand_on_laptop),gesture_labels["palm_up"], gesture_labels["wave"], gesture_labels["slice"], gesture_labels["raise_sway"], gesture_labels["pointing"],pointing_target, bbox_x1, bbox_y1, bbox_x2, bbox_y2])
            
                video_writer.write(draw_frame)
        #except Exception as e:
                #print(f"[Frame {frame_idx}] Tracker error for ID {pid}: {e}")
           # print(f"[Frame {frame_idx}] Tracker error for ID {pid}: {type(e).__name__}: {e}")
            traceback.print_exc()
    frame_idx += 1

    cap.release()
    video_writer.release()
    filtered_persons = {pid: records for pid, records in head_y_by_person.items() if len(records) >= 10}
    nod_frames_by_person = detect_nods_per_person(filtered_persons, fps)
    shake_frames_by_person = detect_shakes_per_person(filtered_persons, fps)
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "timestamp_sec", "person_id",# "head_y", "torso_angle", 
            "x1", "y1", "x2", "y2","posture",
            "smile",'hand_on_laptop', "nodding",'skake_head',"palm_up", "wave", "slice", "raise_sway",'pointing','pointed_object',
            
        ])
        for row in full_log:
            frame = row[0]
            pid = row[2]
            #is_nod = 1 if frame in nod_frames_by_person.get(pid, []) else 0
            is_nod = 1 if any(abs(frame - f) <= 3 for f in nod_frames_by_person.get(pid, [])) else 0
            is_shake = 1 if any(abs(frame - f) <= 3 for f in shake_frames_by_person.get(pid, [])) else 0
            writer.writerow(row[:9] + [is_nod, is_shake] + row[9:])

            #writer.writerow(row + [is_nod,is_shake +gesture_labels["palm_up"], gesture_labels["wave"],gesture_labels["slice"],gesture_labels["raise_sway"],gesture_labels["pointing"]])
        #for obj in object_log:
         #   writer.writerow([""] * 9 + obj)
    with open("trant_obj_mock.csv", mode="w", newline="") as f_obj:
            writer = csv.writer(f_obj)
            writer.writerow(["frame", "timestamp_sec", "class_id",'class_name', "x1", "y1", "x2", "y2", "score"])
            for obj in object_log:
                writer.writerow(obj)

    with open('object_movement_log_mock.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'object_id', 'label_name', 'movement_distance','movement_angle_deg'])
        writer.writerows(object_movement_log)
if __name__ == "__main__":
    process_video_with_transt('mock_test.mp4', "trant_person_mock.csv")
