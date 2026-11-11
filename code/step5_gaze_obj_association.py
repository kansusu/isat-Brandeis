import json
import pandas as pd
import sys
import os
from decimal import Decimal, getcontext

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE — expected GPU")

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"[GPU] Using {torch.cuda.get_device_name(0)}")


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)   # or str(obj) if you prefer exactness
        return super().default(obj)


def find_gaze_target(gaze_x, gaze_y, object_df):
    """
    Checks if a single gaze point (x, y) hits any object in the object_df.
    Returns the class_name of the first object hit, or None.
    """
    if pd.isna(gaze_x) or pd.isna(gaze_y):
        return None
        
    # Find all objects hit by this gaze point
    hits = object_df[
        (gaze_x >= object_df['x1']) & (gaze_x <= object_df['x2']) &
        (gaze_y >= object_df['y1']) & (gaze_y <= object_df['y2'])
    ]
    
    if not hits.empty:
        # Return the class_name of the first object hit
        return hits.iloc[0]['class_name']
        
    return None

def summarize_modalities(posture_data):
    """
    Analyzes the posture data to determine if key social modalities are present.
    If 'skipped' is found for posture/gesture, that is the reported status.
    Returns: A dictionary of modality checks and the calculated metrics.
    """
    
    # 1. Initialize checks and counters
    modality_checks = {
        'lean_in': 'no',
        'pointing_gesture': 'no',
        'gaze_coordinates': 'no',
        'person_to_person_gaze': 'no',
        'joint_attention': 'no',       
        'object_gaze': 'no',
    }
    
    posture_skipped_flag = False
    gesture_skipped_flag = False
    
    # Metrics calculation variables
    num_frames = 0
    total_persons = 0

    # 2. Define known non-person/non-object targets 
    non_specific_targets = {'N/A', 'default', 'out_of_frame'}
    
    # 3. Iterate through data to find any instance of each modality and count people/frames
    for image_name, frame_data in posture_data.items():
        
        # Count frames
        num_frames += 1
        
        # Count persons in this frame and update total_persons
        num_persons_in_frame = len(frame_data)
        total_persons += num_persons_in_frame
        
        for person_id, person_data in frame_data.items():
            
            # --- Check for Posture (Step 2) ---
            posture = person_data.get('posture')
            if posture == 'skipped':
                posture_skipped_flag = True
            elif posture == 'lean-in':
                modality_checks['lean_in'] = 'yes'
                
            # --- Check for Gesture (Step 2) ---
            gesture = person_data.get('gesture')
            if gesture == 'skipped':
                gesture_skipped_flag = True
            elif gesture and gesture != 'none':
                modality_checks['pointing_gesture'] = 'yes'
                
            # --- Check for Gaze Coordinates (Step 3) ---
            gaze_x = person_data.get('gaze_x')
            gaze_y = person_data.get('gaze_y')
            if gaze_x is not None and gaze_y is not None:
                modality_checks['gaze_coordinates'] = 'yes'
                
            # --- Check for Gaze Target (Step 3/5) ---
            gaze_target = person_data.get('gaze_target')
            
            if gaze_target:
                # Check for Person-to-Person Gaze
                if gaze_target.startswith('person_'):
                    modality_checks['person_to_person_gaze'] = 'yes'
                
                # Check for Object Gaze
                elif not gaze_target.startswith('shared_target_cluster') and \
                     not gaze_target.startswith('person_') and \
                     gaze_target not in non_specific_targets:
                    modality_checks['object_gaze'] = 'yes'
                    
            # --- Check for Joint Attention (Step 3) ---
            gaze_type = person_data.get('gaze_attention_type')
            if gaze_type == 'joint_attention':
                 modality_checks['joint_attention'] = 'yes'
        
    # 4. Final adjustments for 'skipped' status
    if posture_skipped_flag:
        modality_checks['lean_in'] = 'skipped'
    if gesture_skipped_flag:
        modality_checks['pointing_gesture'] = 'skipped'
            
    # 5. Calculate Metrics
    if num_frames > 0:
        avg_persons_per_frame = Decimal(total_persons) / Decimal(num_frames)
    else:
        avg_persons_per_frame = Decimal("0")
        
    avg_persons_per_frame = avg_persons_per_frame.quantize(Decimal("0.01"))

    # 6. Format the final summary dictionary
    final_summary = {
        'pipeline_metrics': {
            'total_frames_processed': num_frames,
            'total_person_instances': total_persons,
            'average_persons_per_frame': avg_persons_per_frame
        },
        'modality_presence': {
            'lean_in': modality_checks['lean_in'],
            'pointing_gesture': modality_checks['pointing_gesture'],
            'gaze_coordinates': modality_checks['gaze_coordinates'],
            'person_to_person_gaze': modality_checks['person_to_person_gaze'],
            'joint_attention': modality_checks['joint_attention'], 
            'object_gaze': modality_checks['object_gaze']
        }
    }
    
    # Print a text summary for console/legacy log (optional, but helpful)
    print("\n--- Modality Summary (Text Output) ---")
    print(f"Total Frames Processed: {final_summary['pipeline_metrics']['total_frames_processed']}")
    print(f"Average Persons Per Frame: {final_summary['pipeline_metrics']['average_persons_per_frame']}\n")
    for modality, status in final_summary['modality_presence'].items():
        print(f"'{modality}': '{status}'")
    print("------------------------------------------")
    
    return final_summary


def main(posture_json_path, object_csv_path, summary_json_path):
    # --- 1. Load Posture/Gaze JSON ---
    try:
        with open(posture_json_path, 'r') as f:
            posture_data = json.load(f)
        print(f"[INFO] Loaded posture/gaze data from {posture_json_path}")
    except FileNotFoundError:
        print(f"[ERROR] Could not find posture JSON at {posture_json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] Posture JSON at {posture_json_path} is corrupted.")
        sys.exit(1)

    # --- 2. Load Object CSV ---
    try:
        df_objects = pd.read_csv(object_csv_path)
        print(f"[INFO] Loaded object detections from {object_csv_path}")
        # Group objects by image for faster lookup
        objects_by_image = {image: group for image, group in df_objects.groupby('image')}
    except FileNotFoundError:
        print(f"[ERROR] Could not find object CSV at {object_csv_path}")
        objects_by_image = {}
        print("[INFO] Continuing without object association.")
    except Exception as e:
        print(f"[ERROR] Could not read object CSV: {e}")
        sys.exit(1)


    # --- 3. Process and Update Data (Gaze-to-Object Mapping) ---
    print("[INFO] Associating gaze with objects...")
    updated_frames = 0
    updated_persons = 0

    # Iterate through each frame in the posture JSON
    for image_name, frame_data in posture_data.items():
        if image_name not in objects_by_image:
            continue
            
        objects_in_frame = objects_by_image[image_name]
        frame_was_updated = False
        
        # Iterate through each person in the frame
        for person_id, person_data in frame_data.items():
            gaze_x = person_data.get('gaze_x')
            gaze_y = person_data.get('gaze_y')
            
            if gaze_x is not None and gaze_y is not None:
                target_object_name = find_gaze_target(gaze_x, gaze_y, objects_in_frame)
                
                if target_object_name:
                    original_target = person_data.get('gaze_target', 'N/A')
                    
                    # We will overwrite the gaze_target IF it is NOT already a person-to-person gaze ("person_...").
                    if not original_target.startswith('person_'):
                         person_data['gaze_target'] = target_object_name
                         frame_was_updated = True
                         updated_persons += 1
                    # ALSO update the specific gaze_attention_type if it was a generic cluster
                    if original_target.startswith('shared_target_cluster_'):
                         person_data['gaze_attention_type'] = 'object_gaze'
                    

        if frame_was_updated:
            updated_frames += 1

    print(f"[INFO] Processing complete.")
    print(f"[INFO] Updated gaze target for {updated_persons} person-instances across {updated_frames} frames.")

    # --- 4. Save Updated JSON ---
    try:
        with open(posture_json_path, 'w') as f:
            json.dump(posture_data, f, indent=2)
        print(f"[INFO] Successfully saved updated data back to {posture_json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save updated JSON: {e}")
        
    # --- 5. Generate and Save Modality Summary (NEW STEP) ---
    summary_data = summarize_modalities(posture_data)
    
    try:
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=2, cls=DecimalEncoder)
        print(f"[INFO] Successfully saved summary JSON to {summary_json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save summary JSON: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4: # Changed from 3 to 4 arguments
        print("Usage: python step5-gaze_object.py <path_to_posture.json> <path_to_objects.csv> <path_to_summary.json>")
        sys.exit(1)
        
    main(sys.argv[1], sys.argv[2], sys.argv[3])