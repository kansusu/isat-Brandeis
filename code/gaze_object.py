import json
import pandas as pd
import sys
import os

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

def main(posture_json_path, object_csv_path):
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
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not read object CSV: {e}")
        sys.exit(1)


    # --- 3. Process and Update Data ---
    print("[INFO] Associating gaze with objects...")
    updated_frames = 0
    updated_persons = 0

    # Iterate through each frame in the posture JSON
    for image_name, frame_data in posture_data.items():
        if image_name not in objects_by_image:
            # No objects were detected in this frame, so skip it
            continue
            
        # Get the DataFrame of objects for this specific image
        objects_in_frame = objects_by_image[image_name]
        frame_was_updated = False
        
        # Iterate through each person in the frame
        for person_id, person_data in frame_data.items():
            # Check if gaze_x/gaze_y exists (from Step 3)
            gaze_x = person_data.get('gaze_x')
            gaze_y = person_data.get('gaze_y')
            
            # Only proceed if we have gaze coordinates
            if gaze_x is not None and gaze_y is not None:
                # Find the object (if any) that this person is looking at
                target_object_name = find_gaze_target(gaze_x, gaze_y, objects_in_frame)
                
                if target_object_name:
                    # An object was hit by the gaze.
                    original_target = person_data.get('gaze_target', 'N/A')
                    
                    # --- 🟥 THIS IS THE FIX ---
                    # We will overwrite the gaze_target IF it is NOT
                    # already a person-to-person gaze ("person_...").
                    # This correctly overwrites "shared_target_cluster_0",
                    # "default", and "N/A".
                    if not original_target.startswith('person_'):
                         person_data['gaze_target'] = target_object_name
                         frame_was_updated = True
                         updated_persons += 1
                    # --- 🟥 END FIX ---

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python step5_update_json.py <path_to_posture.json> <path_to_objects.csv>")
        sys.exit(1)
        
    main(sys.argv[1], sys.argv[2])