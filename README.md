# RGB audio-visual perception pipeline: everything in one place

| Folder | What's inside |
| --- | --- |
| `code/` | The five-stage pipeline (`multimodal-attention-analyzer`): frame extraction → person tracking, pose, posture and pointing → gaze → object detection → gaze–object association. Weights and third-party repos (TransT, Detectron2, Gaze-LLE, MMDetection) are not included; see `code/README.md` and `code/scripts/download_weights.sh`. |
| `figures/` | Figures 0.1–0.5 from the chapter, extracted at full resolution. |
| `results/nice/` | Evaluation against expert NICE annotations: per-behavior accuracy and F1, setup, error analysis. |
| `results/tobii/` | Head-based gaze vs. Tobii eye tracking, 12 sessions. `python results/tobii/summarize.py individual_attention` prints sensitivity 0.911, specificity 0.118, balanced accuracy 0.515. |
| `papers/` | The chapter and the AAAI 2025 paper as PDFs, plus `PAPERS.md` listing the other connected papers. |

## Figures

| File | Chapter caption |
| --- | --- |
| `fig0.1_pipeline_flowchart.png` | Flowchart of the detection pipeline |
| `fig0.2_person_detection.png` | The pipeline detects three individuals, each segmented for visualization |
| `fig0.3a/b/c_gaze_scenario.jpg` | Three gaze visualization scenarios |
| `fig0.4a_no_pointing.png`, `fig0.4b_pointing_detected.png` | Pointing detection: (a) no pointing, (b) a pointing event |
| `fig0.5_laptop_rotation.png` | The laptop rotated toward the group; box and motion trace |

## Models used (from the code)

| Stage | Model and setting |
| --- | --- |
| Person detection | Detectron2 Keypoint R-CNN R50-FPN 3x (COCO), score ≥ 0.5, first frame only |
| Tracking | TransT, one tracker per person |
| Pose | MMPose RTMPose-m (SimCC), 256×192, 17 COCO keypoints |
| Gaze | Gaze-LLE `gazelle_dinov2_vitl14_inout`; head box from keypoints 0–4 |
| Objects | MMDetection RTMDet-L (COCO), laptop class, score ≥ 0.3 |
| Speech (not in this code) | pyannote, Whisper, TalkNet |

## Privacy

The figures come from the chapter as prepared for publication. Raw study recordings are IRB-protected and are not included. Check with the protocol before sharing anything here outside the lab.
