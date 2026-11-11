"""
SQS-driven runner for the multimodal attention pipeline.
Listens to the same SQS message shape used by face-blur-processor, downloads
the referenced blurred chunk from S3, runs the pipeline, uploads per-frame
attention visuals to S3, and writes the attention JSON to DynamoDB.
"""
import glob
import re
import json
import os
import shutil
import sys
import subprocess
import tempfile
import time
import traceback
from typing import Any, Dict, List, Tuple
from step5_gaze_obj_association import summarize_modalities
from decimal import Decimal

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import dotenv_values
from PIL import Image

from run_pipeline import run_pipeline as execute_pipeline, load_config as load_pipeline_config_file

# Load env values
ENV = dotenv_values(".env")

# Resolve queue and bucket settings
QUEUE_URL = ENV.get("MULTIMODAL_SQS_QUEUE_URL") or ENV.get("SQS_QUEUE_URL")
INPUT_BUCKET_DEFAULT = ENV.get("INPUT_BUCKET") or ENV.get("BUCKET")
AWS_REGION = ENV.get("AWS_REGION")
DYNAMO_TABLE = ENV.get("DYNAMO_TABLE")  # optional status table
PIPELINE_CONFIG_FILE = ENV.get("PIPELINE_CONFIG_FILE") or "config.env"
STAGE = ENV.get("STAGE", "dev")
# METRICS_TABLE_NAME = "RecordingChunkMetrics" if STAGE == "prod" else "dev-RecordingChunkMetrics"
METRICS_TABLE_NAME = "RecordingChunkMetrics" 

if not QUEUE_URL:
    raise SystemExit("SQS queue URL not configured (MULTIMODAL_SQS_QUEUE_URL or SQS_QUEUE_URL)")

# Boto clients
boto_kwargs = {"region_name": AWS_REGION} if AWS_REGION else {}
sqs = boto3.client("sqs", **boto_kwargs)
s3 = boto3.client("s3", **boto_kwargs)
dynamo_resource = boto3.resource("dynamodb", **boto_kwargs) if (DYNAMO_TABLE or METRICS_TABLE_NAME) else None
dynamo_status = dynamo_resource.Table(DYNAMO_TABLE) if (dynamo_resource and DYNAMO_TABLE) else None
metrics_table = dynamo_resource.Table(METRICS_TABLE_NAME) if (dynamo_resource and METRICS_TABLE_NAME) else None

REQUIRED_CFG = [
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
]


def sanitize_file_name(file_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "." else "_" for ch in file_name)


def load_pipeline_config():
    cfg = load_pipeline_config_file(PIPELINE_CONFIG_FILE)
    missing = [k for k in REQUIRED_CFG if k not in cfg or not cfg[k]]
    if missing:
        raise SystemExit(f"Missing required keys in {PIPELINE_CONFIG_FILE}: {', '.join(missing)}")
    return cfg


def clear_path(path: str):
    if not path:
        return
    if os.path.isfile(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except FileNotFoundError:
                    pass
            for name in dirs:
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)
    else:
        os.makedirs(path, exist_ok=True)


def reset_outputs(cfg):
    for f in (cfg.get("HEADBOX_JSON"), cfg.get("POSTURE_JSON"), cfg.get("OBJECT_CSV")):
        clear_path(f)
    for d in (cfg.get("FRAME_FOLDER"), cfg.get("STEP2_VISUALS"), cfg.get("STEP3_GAZE_VISUALS")):
        clear_path(d)


def resolve_root_path(file_key: str) -> str:
    if "/Chunks/" in file_key:
        return file_key.split("/Chunks/")[0]
    return os.path.dirname(file_key)


# def parse_media(body: dict):
#     blurred_media = body.get("media") or {}
#     media = body.get("original_media", {}) or blurred_media
#     bucket = media.get("s3_bucket") or INPUT_BUCKET_DEFAULT
#     key = media.get("s3_key")

#     if not key and media.get("media_uri", "").startswith("s3://"):
#         uri = media["media_uri"][5:]
#         parts = uri.split("/", 1)
#         if len(parts) == 2:
#             if not bucket:
#                 bucket = parts[0]
#             key = parts[1]

#     mime = blurred_media.get("mime_type") or body.get("contentType", "") or ""
#     return bucket, key, mime

def parse_media(body: dict):
    """
    Processing media -> body['media'] (blurred ONLY)
    Upload / metrics -> body['original_media']
    """
    blurred = body.get("media")
    original = body.get("original_media")

    if not blurred:
        print("ERROR: No blurred media found in SQS payload")
        raise ValueError("Blurred media missing; refusing to process original media")

    # ---- Blurred media (for processing) ----
    proc_bucket = blurred.get("s3_bucket") or INPUT_BUCKET_DEFAULT
    proc_key = blurred.get("s3_key")

    if not proc_key and blurred.get("media_uri", "").startswith("s3://"):
        uri = blurred["media_uri"][5:]
        proc_bucket, proc_key = uri.split("/", 1)

    # ---- Original media (for uploads / metrics) ----
    orig_bucket = None
    orig_key = None

    if original:
        orig_bucket = original.get("s3_bucket")
        orig_key = original.get("s3_key")

        if not orig_key and original.get("media_uri", "").startswith("s3://"):
            uri = original["media_uri"][5:]
            orig_bucket, orig_key = uri.split("/", 1)

    mime = blurred.get("mime_type", "")

    return proc_bucket, proc_key, orig_bucket, orig_key, mime


def ensure_png(src_path: str) -> Tuple[str, str]:
    """Return a PNG path for upload; convert if needed. Returns (path, tmp_to_cleanup_or_None)."""
    if src_path.lower().endswith(".png"):
        return src_path, None
    img = Image.open(src_path).convert("RGB")
    tmp_path = tempfile.mktemp(suffix=".png")
    img.save(tmp_path, "PNG")
    return tmp_path, tmp_path


def collect_attention_images(step2_dir: str, step3_dir: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Collect attention visual files and normalize names/extensions.
    Returns ([(target_name, local_path), ...], [temps_to_delete]).
    """
    uploads: List[Tuple[str, str]] = []
    temps: List[str] = []

    def add_file(src_path: str, target_name: str):
        png_path, tmp = ensure_png(src_path)
        if tmp:
            temps.append(tmp)
        uploads.append((target_name, png_path))

    # Base visuals (step 2) - expect frame_XXXXX.png style, skip gaze overlays
    if step2_dir and os.path.isdir(step2_dir):
        for path in sorted(glob.glob(os.path.join(step2_dir, "*"))):
            fname = os.path.basename(path)
            if "_gaze" in fname:
                continue
            stem, _ = os.path.splitext(fname)
            add_file(path, f"{stem}.png")

    # Gaze visuals (step 3) - expect *_gaze.*
    if step3_dir and os.path.isdir(step3_dir):
        for path in sorted(glob.glob(os.path.join(step3_dir, "*"))):
            fname = os.path.basename(path)
            stem, _ = os.path.splitext(fname)
            base = stem[:-5] if stem.endswith("_gaze") else stem
            add_file(path, f"{base}_gaze.png")

    return uploads, temps

def extract_chunk_id(s3_key: str) -> str:
    """
    Extract ChunkXXXX from an S3 key or filename.
    Example:
    Dev-Teacher_..._Chunk0003.webm -> Chunk0003
    """
    filename = os.path.basename(s3_key)
    match = re.search(r"(Chunk\d+)", filename, re.IGNORECASE)
    return match.group(1) if match else "UnknownChunk"

def upload_attention_images(bucket: str, prefix: str, files: List[Tuple[str, str]], chunk_id: str) -> Dict[str, str]:
    uploaded = {}
    for target_name, local_path in files:
        final_name = f"{chunk_id}_{target_name}"
        key = f"{prefix}/{final_name}"
        print(f"Uploading {local_path} -> s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)
        uploaded[target_name] = f"s3://{bucket}/{key}"
    return uploaded


# def build_attention_payload(posture_json_path: str) -> Dict[str, Any]:
#     with open(posture_json_path, "r") as f:
#         posture_data = json.load(f)

#     if isinstance(posture_data, dict) and "frames" in posture_data:
#         payload = posture_data
#     else:
#         payload = {"summary": summarize_modalities(posture_data), "frames": posture_data}

#     return {"attention": payload}

def build_attention_payload(
    posture_json_path: str,
    summary_json_path: str,
    chunk_id: str
) -> Dict[str, Any]:

    with open(posture_json_path, "r") as f:
        posture_data = json.load(f)

    with open(summary_json_path, "r") as f:
        summary_data = json.load(f, parse_float=Decimal)
        
    frames_payload = []

    for frame_name, persons in posture_data.items():
        base_frame_id = os.path.splitext(frame_name)[0]
        frame_id = f"{chunk_id}_{base_frame_id}"

        frame_entry = {
            "frame_id": frame_id,
            "visualizations": {
                "posture": f"{frame_id}_posture.png",
                "gaze": f"{frame_id}_gaze.png",
            },
            "people": []
        }

        for person_id, pdata in persons.items():
            frame_entry["people"].append({
                "person_num": person_id,
                "posture": pdata.get("posture"),
                "gesture": pdata.get("gesture"),
                "gaze_attention_type": pdata.get("gaze_attention_type"),
                "gaze_target": pdata.get("gaze_target"),
                "gaze_x": pdata.get("gaze_x"),
                "gaze_y": pdata.get("gaze_y"),
            })

        frames_payload.append(frame_entry)

    return {
        "multimodal_attention": {
            "pipeline_metrics": summary_data["pipeline_metrics"],
            "summary_data": summary_data["modality_presence"],
            "frames": frames_payload
        }   
    }


def record_status(job_id: str, status: str, info: dict):
    if not dynamo_status:
        return
    item = {
        "job_id": job_id,
        "status": status,
        "timestamp": int(time.time()),
        "info": info or {},
    }
    dynamo_status.put_item(Item=item)


def write_metrics(recording_id: str, chunk_key: str, bucket: str, prefix: str, attention_payload: Dict[str, Any]):
    """
    Upsert an entry in dev-RecordingChunkMetrics.
    Key schema: partition key = recordingId, sort key = chunk_media_url.
    """
    if not metrics_table:
        print("Metrics table not configured; skipping Dynamo write.")
        return

    chunk_media_url = (
        f"https://{bucket}.s3.amazonaws.com/{chunk_key}"
    )
    
    key = {
        "recordingId": recording_id,
        "chunk_media_url": chunk_media_url,
    }

    print(f"Updating metrics entry for recordingId = {recording_id}")

    update_expr = "SET multimodal_attention = :mma"

    expr_values = {
        ":mma": attention_payload["multimodal_attention"],  
    }

    try:
        metrics_table.update_item(
            Key=key,
            UpdateExpression=update_expr,
            ExpressionAttributeValues=expr_values,
            ConditionExpression="attribute_exists(recordingId) AND attribute_exists(chunk_media_url)",
        )
        print("----Metrics entry updated successfully for recordingId =", recording_id)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            print(
                f"Entry does not exist for recordingId={recording_id}, "
                f"chunk={chunk_media_url}. Skipping update!"
            )
        else:
            raise



def process_message(msg):
    cfg = load_pipeline_config()
    body = json.loads(msg["Body"])
    proc_bucket, proc_key, orig_bucket, orig_key, mime = parse_media(body)

    if not proc_bucket or not proc_key:
        print("Skipping message: missing bucket/key")
        return
    if "audio" in mime.lower():
        print(f"Skipping audio payload: {mime}")
        return

    job_id = body.get("recordingId") or os.path.splitext(os.path.basename(orig_key))[0]
    chunk_id = extract_chunk_id(proc_key)
    root_path = resolve_root_path(orig_key)
    attention_prefix = f"{root_path}/analysis/multimodal_attention"
    target_bucket = orig_bucket # Upload back to original bucket

    os.makedirs("incoming_videos", exist_ok=True)

    local_file = os.path.join(
    "incoming_videos",
    sanitize_file_name(os.path.basename(proc_key))
    )

    print(f"Downloading s3://{proc_bucket}/{proc_key} -> {local_file}")
    s3.download_file(proc_bucket, proc_key, local_file)

    temps: List[str] = []
    try:
        reset_outputs(cfg)
        record_status(job_id, "running", {"input": f"s3://{proc_bucket}/{proc_key}"})
        # local_file = "KevinGrady_25-01-16_OM_L3_C3_678965a1c29ba000084f1821_Chunk0071_blurred.mp4"
        pipeline_result = execute_pipeline(local_file, PIPELINE_CONFIG_FILE)

        images, temps = collect_attention_images(
            pipeline_result["outputs"]["step2_visuals"],
            pipeline_result["outputs"]["step3_visuals"],
        )
        uploaded = upload_attention_images(target_bucket, attention_prefix, images, chunk_id)

        # attention_payload = build_attention_payload(pipeline_result["outputs"]["posture"])
        attention_payload = build_attention_payload(
            posture_json_path=pipeline_result["outputs"]["posture"], 
            summary_json_path=pipeline_result["outputs"]["summary"],
            chunk_id=chunk_id
        )
        
        if orig_key and "/fixedChunks/" in orig_key:
            orig_key = orig_key.replace("/fixedChunks/", "/Chunks/")

        write_metrics(job_id, orig_key, orig_bucket, attention_prefix, attention_payload)

        record_status(job_id, "success", {"attention_prefix": attention_prefix})
        print(f"\n----- Run completed for recordingId = {job_id} ----- \n Uploaded {len(uploaded)} attention images to s3://{target_bucket}/{attention_prefix}")
    except (subprocess.CalledProcessError, BotoCoreError, ClientError, OSError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        err_info = {"error": str(e), "trace": traceback.format_exc()}
        record_status(job_id, "failed", err_info)
        print(f"Job {job_id} failed: {e}")
        traceback.print_exc()
    finally:
        for tmp in temps:
            try:
                print("Cleaning up temp file:", tmp)
                os.remove(tmp)
            except FileNotFoundError:
                print("Temp file not found for cleanup:", tmp)
                pass
        try:
            print("Cleaning up downloaded file:", local_file)
            os.remove(local_file)
        except FileNotFoundError:
            print("Downloaded file not found for cleanup:", local_file)
            pass


def main():
    print("Starting Multimodal SQS listener...")
    print(f"Listening to SQS Queue: {QUEUE_URL}")
    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10,
                VisibilityTimeout=1800
            )
            for msg in resp.get("Messages", []):
                try:
                    process_message(msg)
                except Exception as e:
                    print(f"Error processing message: {e}")
                    traceback.print_exc()
                finally:
                    sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])
                    print("Message deleted from SQS...")
        except KeyboardInterrupt:
            print("Exiting on user interrupt")
            break
        except Exception as e:
            print(f"SQS polling error: {e}")
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
