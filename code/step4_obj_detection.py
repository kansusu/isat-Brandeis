import cv2
import numpy as np
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from run_pipeline import log_device
import sys
import os
import time  # NEW: Import the time module
try:
    from tqdm import tqdm
except ImportError:
    print("Note: 'tqdm' not found. Progress bar will not be shown.")
    print("Install with: pip install tqdm")
    # Create a simple wrapper if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable
    
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE — expected GPU")

_orig_torch_load = torch.load
def _torch_load_no_weights_only(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_no_weights_only

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"[GPU] Using {torch.cuda.get_device_name(0)}")


# --- Constants ---
# This is the 'laptop' class ID from the standard COCO dataset.
# *** You MUST change this if your model uses a different class ID! ***
LAPTOP_CLASS_ID = 63 

# Score threshold from your original script
SCORE_THRESH = 0.30

# CHANGED: Added supported image extensions
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def main(image_folder, config_file, checkpoint_file, output_csv):
    """
    Detects ALL laptops in a folder of images and saves their
    bounding box coordinates (x1, y1, x2, y2) to a CSV.
    Also reports processing time.
    """
    # NEW: Start total script timer
    script_start_time = time.time()
    
    # --- 1. Input Validation ---
    # CHANGED: Check if it's a directory
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        return
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return
    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint file not found at {checkpoint_file}")
        return

    # --- 2. Initialize Model ---
    print("Initializing detector...")
    model_load_start = time.time() # NEW
    
    # Use 'cuda:0' if you have a GPU, 'cpu' otherwise
    device = 'cuda:0'
    log_device(f"Step 4 detector -> {device}")
    try:
        model = init_detector(config_file, checkpoint_file, device=device)
        
        # === THIS LINE IS THE FIX ===
        # MMDetection 3.x stores class names in 'dataset_meta'
        class_names = model.dataset_meta['classes']
        # ===========================
        
        laptop_class_name = class_names[LAPTOP_CLASS_ID]
        model_load_end = time.time() # NEW
        print(f"Detector initialized in {model_load_end - model_load_start:.2f}s. Looking for class ID {LAPTOP_CLASS_ID} ('{laptop_class_name}').")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        print("Please ensure MMDetection and its dependencies are installed correctly.")
        print("Also check that your model's LAPTOP_CLASS_ID is correct.")
        return

    # --- 3. Open Video File ---
    # CHANGED: This section now finds image files instead of opening a video
    print(f"Scanning image folder: {image_folder}")
    image_paths = []
    for fname in os.listdir(image_folder):
        if fname.lower().endswith(SUPPORTED_EXTENSIONS):
            image_paths.append(os.path.join(image_folder, fname))
    
    if not image_paths:
        print(f"Error: No images with extensions {SUPPORTED_EXTENSIONS} found in {image_folder}")
        return
        
    total_images = len(image_paths)
    print(f"Found {total_images} images.")
    # ---

    detection_results = []

    # --- 4. Process Video ---
    # CHANGED: Loop over image paths instead of reading video frames
    print("Processing images...")
    
    # NEW: Start inference timer
    inference_start_time = time.time()
    
    pbar = tqdm(image_paths, desc="Processing images")
    for image_path in pbar:
        # CHANGED: Read image file
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # Run detection (THIS IS THE SAME AS BEFORE)
        result = inference_detector(model, frame)

        # Extract detections (THIS IS THE SAME AS BEFORE)
        # This checks for MMDetection 3.x API
        if hasattr(result, 'pred_instances'):
            bboxes = result.pred_instances.bboxes.cpu().numpy()
            scores = result.pred_instances.scores.cpu().numpy()
            labels = result.pred_instances.labels.cpu().numpy()
        else:
            # --- MMDetection 2.x API Logic ---
            all_bboxes = []
            all_scores = []
            all_labels = []
            if isinstance(result, list):
                for class_id, class_detections in enumerate(result):
                    if class_detections.shape[0] > 0:
                        all_bboxes.append(class_detections[:, :4])
                        all_scores.append(class_detections[:, 4])
                        all_labels.append(np.full(class_detections.shape[0], class_id))
                
                if not all_labels: # No detections at all
                    bboxes, scores, labels = np.empty((0, 4)), np.array([]), np.array([])
                else:
                    bboxes = np.concatenate(all_bboxes)
                    scores = np.concatenate(all_scores)
                    labels = np.concatenate(all_labels)
            else:
                print(f"Warning: Unexpected detection result format in image {image_path}.")
                continue


        # Filter by class ID and score (THIS IS THE SAME AS BEFORE)
        laptop_indices = (labels == LAPTOP_CLASS_ID) & (scores >= SCORE_THRESH)
        laptop_boxes = bboxes[laptop_indices]
        laptop_scores = scores[laptop_indices]

        # Format frame name
        # CHANGED: Use the actual image filename
        image_name = os.path.basename(image_path)

        # *** Loop over all detected laptops and save Bounding Box *** (THIS IS THE SAME AS BEFORE)
        if len(laptop_boxes) > 0:
            for i in range(len(laptop_boxes)):
                box = laptop_boxes[i]
                score = laptop_scores[i]
                
                x1, y1, x2, y2 = box

                detection_results.append({
                    "image": image_name,
                    "class_name": laptop_class_name,
                    "score": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
        

    # NEW: Stop inference timer
    inference_end_time = time.time()
    total_inference_time = inference_end_time - inference_start_time
    
    print("Image processing complete.")

    # --- 5. Save Results --- (THIS IS THE SAME AS BEFORE)
    if not detection_results:
        print(f"Warning: No laptops were detected in any images with threshold {SCORE_THRESH}.")
        # Create an empty file with correct headers
        df = pd.DataFrame(columns=["image", "class_name", "score", "x1", "y1", "x2", "y2"])
    else:
        df = pd.DataFrame(detection_results)
        
    df.to_csv(output_csv, index=False)
    print(f"Detection results saved to {output_csv}")

    # --- NEW: Timing Report ---
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    
    print("\n--- Processing Time Report ---")
    print(f"Total script runtime:   {total_script_time:.2f} seconds")
    print(f"Total images processed: {total_images}")
    print(f"Total inference time:   {total_inference_time:.2f} seconds (excludes model loading)")
    
    if total_images > 0 and total_inference_time > 0:
        avg_time_per_image = total_inference_time / total_images
        fps = total_images / total_inference_time
        print(f"Average time per image: {avg_time_per_image:.4f} seconds")
        print(f"Frames Per Second (FPS): {fps:.2f}")
    # --- End of NEW section ---

if __name__ == "__main__":
    if len(sys.argv) != 5:
        # CHANGED: Updated usage example
        print("\nUsage: python detect_all_laptop_boxes_v2.py <image_folder_path> <config_file> <checkpoint_file> <output_csv_path>")
        print("Example: python detect_all_laptop_boxes_v2.py ./my_images mmdet.py model.pth all_laptop_boxes.csv\n")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])