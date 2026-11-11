import cv2
import time
import sys
import os # Import the os module for directory creation and path joining

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE — expected GPU")

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"[GPU] Using {torch.cuda.get_device_name(0)}")


def process_frame(frame):
    """
    Put all your model inference logic here (YOLO, Gazelle, etc.)
    This function takes a frame and returns the processed frame.
    """
    # For now, it just returns the frame.
    # Replace this with your actual model code.
    processed_frame = frame
    return processed_frame

# --- MODIFIED FUNCTION SIGNATURE ---
def run_video_processing(video_source_path, output_folder_path, frame_skip_rate):
    """
    Main loop to read a video, process frames, and save each frame
    as an individual image file in the specified folder.
    No GUI functions are used.
    """
    
    # --- 1. Initialize VideoCapture ---
    cap = cv2.VideoCapture(video_source_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source_path}'")
        return

    # --- 2. Get Video Properties (for info) ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if original_fps <= 0:
        print("Warning: Could not get original FPS. Defaulting to 30.")
        original_fps = 30

    print(f"Input video: {frame_width}x{frame_height} @ {original_fps:.2f} FPS")

    # --- 3. Create Output Directory ---
    # This will create the folder if it doesn't exist and won't crash if it does.
    try:
        os.makedirs(output_folder_path, exist_ok=True)
        print(f"Outputting frames to folder: {output_folder_path}")
    except OSError as e:
        print(f"Error: Could not create output directory '{output_folder_path}': {e}")
        cap.release()
        return

    # --- VideoWriter (REMOVED) ---
    # We are no longer writing to a video file.

    prev_time = 0
    frame_count = 0
    saved_frame_count = 0 # Added a counter for saved frames

    # --- 4. Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # --- MODIFIED PRINT ---
            print(f"\nFinished processing. Total frames read: {frame_count}. Total frames saved: {saved_frame_count}")
            break

        # --- A. Process Frame ---
        frame = process_frame(frame)

        # --- B. Calculate & Display FPS ---
        current_time = time.time()
        if (current_time - prev_time) > 0:
            fps = 1 / (current_time - prev_time)
        else:
            fps = 0 # (for the first frame)
        prev_time = current_time
        frame_count += 1

        # --- C. Add FPS text to the frame ---
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- D. Write the frame to an image file (MODIFIED) ---
        
        # --- NEW: Conditional save logic ---
        # We check if the frame_count is a multiple of frame_skip_rate
        if frame_count % frame_skip_rate == 0:
            # We use :06d to pad the number with zeros (e.g., 000001, 000002)
            # Using frame_count in the name ensures it matches the video frame number
            filename = f"frame_{frame_count:06d}.png"
            output_file_path = os.path.join(output_folder_path, filename)
            
            try:
                cv2.imwrite(output_file_path, frame)
                saved_frame_count += 1 # Increment the save counter
                # Update console progress
                print(f"Saved: {filename} (Frame {frame_count}) | Processing FPS: {fps:.2f}", end='\r')
            except Exception as e:
                print(f"\nError writing frame {frame_count}: {e}")
                break
        else:
            # --- NEW: Print progress for skipped frames ---
            print(f"Skipped: Frame {frame_count} | Processing FPS: {fps:.2f}", end='\r')


    # --- 5. Clean up ---
    print("\nReleasing video source...")
    cap.release()
    # No writer to release
    print("Done.")

# --- This is the "script" part (MODIFIED) ---
if __name__ == "__main__":
    
    # --- MODIFIED: Check for 4 arguments ---
    if len(sys.argv) < 4:
        print("Usage: python process_video_headless.py <input_video_path> <output_folder_path> <frame_skip_rate>")
        print("Example: python process_video_headless.py input.mp4 ./frames_output/ 5")
        print("         (This will save one frame every 5 frames)")
        sys.exit(1) # Exit with an error code
    
    source_path = sys.argv[1]
    output_folder = sys.argv[2]
    
    # --- NEW: Parse frame_skip_rate with error handling ---
    try:
        frame_skip_rate = int(sys.argv[3])
        if frame_skip_rate <= 0:
            print("Error: Frame skip rate must be a positive integer (1 or greater).")
            sys.exit(1)
    except ValueError:
        print(f"Error: Frame skip rate must be an integer. You provided: '{sys.argv[3]}'")
        sys.exit(1)
    # --- End new argument parsing ---
    
    print(f"Starting video processing...")
    print(f"Input file:   {source_path}")
    print(f"Output folder: {output_folder}")
    # --- NEW: Print the skip rate ---
    print(f"Saving 1 frame every {frame_skip_rate} frames.")
    
    # --- MODIFIED: Pass the new argument to the function ---
    run_video_processing(source_path, output_folder, frame_skip_rate)