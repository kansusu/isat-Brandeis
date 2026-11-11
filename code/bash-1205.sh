#!/bin/bash
#
# This script runs a 5-step computer vision pipeline.
# Configuration variables are loaded from the 'config.env' file.
# The input video path MUST be provided as the first command-line argument.
#

# --- Stop the script if any command fails ---
set -e

# -----------------------------------------------
# --- 1. CONFIGURATION LOADING & ARGUMENT PARSING ---
# -----------------------------------------------

CONFIG_FILE="config-1205.env"

# Load variables from the config.env file
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "--- !!! ERROR !!! ---"
    echo "Configuration file '$CONFIG_FILE' not found."
    echo "Please create a '$CONFIG_FILE' file with the necessary variables."
    exit 1
fi

# Check for the required command-line argument (Input Video)
if [ -z "$1" ]; then
    echo "--- !!! ERROR !!! ---"
    echo "Usage: $0 <INPUT_VIDEO_PATH>"
    echo "Please provide the path to the input video as the first argument."
    exit 1
fi

# The input video is now passed as the first argument
INPUT_VIDEO="$1"
echo "Input Video set to: $INPUT_VIDEO"
echo "Frame Skip Rate set to: $FRAME_SKIP_RATE"

# -----------------------------------------------
# --- 2. PIPELINE SETUP & RUN ---
# -----------------------------------------------

# --- Initialize timing variables ---
PIPELINE_START=$(date +%s)
LAST_TIME=$PIPELINE_START
STEP1_TIME=0
STEP2_TIME=0
STEP3_TIME=0
STEP4_TIME=0
STEP5_TIME=0


echo "--- Starting Pipeline ---"

# --- Setup: Create output directories ---
mkdir -p "$FRAME_FOLDER"
mkdir -p "$STEP2_VISUALS"
mkdir -p "$STEP3_GAZE_VISUALS"
# Create an empty file for the posture JSON
touch "$POSTURE_JSON"
echo "{}" > "$POSTURE_JSON"
# Delete any old headbox file to ensure Step 2 creates it
rm -f "$HEADBOX_JSON"

# --- NEW VARIABLE for Step 5 log ---
STEP5_LOG="step5_summary_report.log"

# ------------------------------------------------
# --- [Step 1/5] Extracting frames ---
# ------------------------------------------------
echo ""
echo "--- [Step 1/5] Extracting frames from $INPUT_VIDEO ---"
python extract.py "$INPUT_VIDEO" "$FRAME_FOLDER" "$FRAME_SKIP_RATE"

# --- Calculate Step 1 Time ---
CURRENT_TIME=$(date +%s)
STEP1_TIME=$((CURRENT_TIME - LAST_TIME))
LAST_TIME=$CURRENT_TIME
echo "--- Step 1 Complete (Time: ${STEP1_TIME}s) ---"


# --- ERROR CHECKING FOR STEP 1 ---
echo "Checking output of Step 1..."
if [ -z "$(ls -A "$FRAME_FOLDER")" ]; then
    echo ""
    echo "--- !!! PIPELINE FAILED AT STEP 1 !!! ---"
    echo "Error: Step 1 did NOT extract any frames to '$FRAME_FOLDER'."
    echo "  Please verify: 1) INPUT_VIDEO path: '$INPUT_VIDEO' 2) Video integrity."
    exit 1
fi
echo "Listing a sample of created frames:"
ls -l "$FRAME_FOLDER" | tail -n 5
echo "[OK] Step 1 output (frames) validated."


# ------------------------------------------------
# --- [Step 2/5] Detecting headboxes and posture/non-verbal ---
# ------------------------------------------------
echo ""
echo "--- [Step 2/5] Detecting headboxes and posture/non-verbal (Mode: $STEP2_MODE) ---"
echo "(Creating headbox file: $HEADBOX_JSON)"
echo "(Creating posture file: $POSTURE_JSON)"
python "step2-1205.py" \
  --input_folder "$FRAME_FOLDER" \
  --output_headbox_json "$HEADBOX_JSON" \
  --output_visual_folder "$STEP2_VISUALS" \
  --output_posture_json "$POSTURE_JSON" \
  --mode "$STEP2_MODE" # <--- NEW ARGUMENT

# --- Calculate Step 2 Time ---
CURRENT_TIME=$(date +%s)
STEP2_TIME=$((CURRENT_TIME - LAST_TIME))
LAST_TIME=$CURRENT_TIME
echo "--- Step 2 Complete (Time: ${STEP2_TIME}s) ---"


# --- ERROR CHECKING FOR STEP 2 ---
if [ ! -f "$HEADBOX_JSON" ]; then
    echo "--- !!! PIPELINE FAILED AT STEP 2 !!! ---"
    echo "Error: Step 2 did NOT create the headbox file: $HEADBOX_JSON"
    exit 1
fi

if [ ! -s "$POSTURE_JSON" ] || [ "$(cat $POSTURE_JSON)" = "{}" ]; then
    # Note: This check might fail if STEP2_MODE="headbox_only", but subsequent steps rely on the file existing.
    # The Python script now ensures it at least creates an empty file.
    echo "[Warning] Step 2 Posture JSON ($POSTURE_JSON) is empty, likely due to STEP2_MODE='headbox_only' or no detections."
fi
echo "[OK] Step 2 output files validated (Headbox JSON exists)."


# ------------------------------------------------
# --- [Step 3/5] Inferring gaze (Gaze-Only) ---
# ------------------------------------------------
echo ""
echo "--- [Step 3/5] Inferring gaze (Gaze-Only) ---"
echo "(Reading $HEADBOX_JSON, Updating $POSTURE_JSON)"
python step3-1205.py \
  --gazelle_checkpoint "$GAZELLE_CHECKPOINT" \
  --image_folder "$FRAME_FOLDER" \
  --head_box_json "$HEADBOX_JSON" \
  --posture_json "$POSTURE_JSON" \
  --output_dir "$STEP3_GAZE_VISUALS"

# --- Calculate Step 3 Time ---
CURRENT_TIME=$(date +%s)
STEP3_TIME=$((CURRENT_TIME - LAST_TIME))
LAST_TIME=$CURRENT_TIME
echo "--- Step 3 Complete (Time: ${STEP3_TIME}s) ---"


# ------------------------------------------------
# --- [Step 4/5] Detecting laptops/objects ---
# ------------------------------------------------
echo ""
echo "--- [Step 4/5] Detecting laptops/objects ---"
echo "(Creating the object CSV: $OBJECT_CSV)"
python laptop-1108.py \
  "$FRAME_FOLDER" \
  "$MMDET_CONFIG" \
  "$MMDET_CHECKPOINT" \
  "$OBJECT_CSV"

# --- Calculate Step 4 Time ---
CURRENT_TIME=$(date +%s)
STEP4_TIME=$((CURRENT_TIME - LAST_TIME))
LAST_TIME=$CURRENT_TIME
echo "--- Step 4 Complete (Time: ${STEP4_TIME}s) ---"


# ------------------------------------------------
# --- [Step 5/5] Associating gaze and objects ---
# ------------------------------------------------
echo ""
echo "--- [Step 5/5] Associating gaze and objects ---"
echo "(Reading $POSTURE_JSON and $OBJECT_CSV, Updating $POSTURE_JSON)"
# --- MODIFIED: Redirect output to STEP5_LOG ---
python step5-1205.py \
  "$POSTURE_JSON" \
  "$OBJECT_CSV" > "$STEP5_LOG"

# --- Calculate Step 5 Time ---
CURRENT_TIME=$(date +%s)
STEP5_TIME=$((CURRENT_TIME - LAST_TIME))
LAST_TIME=$CURRENT_TIME
echo "--- Step 5 Complete (Time: ${STEP5_TIME}s) ---"


echo ""
echo "--- Pipeline Finished ---"

# ------------------------------------------------
# --- 3. FINAL REPORT ---
# ------------------------------------------------

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - PIPELINE_START))

# Format all times for the report
STEP1_F=$(printf '%02dm %02ds' $((STEP1_TIME / 60)) $((STEP1_TIME % 60)))
STEP2_F=$(printf '%02dm %02ds' $((STEP2_TIME / 60)) $((STEP2_TIME % 60)))
STEP3_F=$(printf '%02dm %02ds' $((STEP3_TIME / 60)) $((STEP3_TIME % 60)))
STEP4_F=$(printf '%02dm %02ds' $((STEP4_TIME / 60)) $((STEP4_TIME % 60)))
STEP5_F=$(printf '%02dm %02ds' $((STEP5_TIME / 60)) $((STEP5_TIME % 60)))
TOTAL_F=$(printf '%02dm %02ds' $((TOTAL_SECONDS / 60)) $((TOTAL_SECONDS % 60)))

echo "================================================="
echo "               Pipeline Runtime Report"
echo "================================================="
echo "Step 1: Extract Frames    ... $STEP1_F  ($STEP1_TIME seconds)"
echo "Step 2: Posture/Non-Verbal ... $STEP2_F  ($STEP2_TIME seconds)"
echo "Step 3: Gaze Inference    ... $STEP3_F  ($STEP3_TIME seconds)"
echo "Step 4: Object Detection  ... $STEP4_F  ($STEP4_TIME seconds)"
echo "Step 5: Gaze/Object Assoc ... $STEP5_F  ($STEP5_TIME seconds)"
echo "-------------------------------------------------"
echo "Total Pipeline Runtime: $TOTAL_F ($TOTAL_SECONDS seconds)"
echo "================================================="

echo ""
echo "================================================="
echo "          FINAL MODALITY SUMMARY (from Step 5)"
echo "================================================="
# --- Print the content of the log file ---
cat "$STEP5_LOG" 
echo "================================================="

echo "Final Updated JSON: $POSTURE_JSON"
echo "Object CSV (from Step 4): $OBJECT_CSV"