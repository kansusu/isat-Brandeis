<div align="center">

# 👀 Nonverbal Behavior Recognition from RGB Video

**A five-stage perception pipeline that detects gaze, posture and pointing in multiparty collaboration, from a single camera.**

![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![AAAI 2025](https://img.shields.io/badge/AAAI-2025-6250d6)

<img src="figures/fig0.1_pipeline_flowchart.png" alt="Pipeline flowchart" width="90%">

</div>

---

## ✨ At a glance

| | |
| :-- | :-- |
| 🎯 **NICE overall** | accuracy **0.73** · F1 **0.59** over 12 expert-annotated clips |
| 🏆 **Best behavior** | *Look away*: accuracy 0.96 · F1 0.82 |
| 👁️ **vs. Tobii eye tracking** | mean F1 **72.7%** (individual) · **68.4%** (group), 12 sessions |
| ⚠️ **Key limitation** | specificity **0.118**: facing the screen ≠ looking at it |

## 🔧 How it works

```mermaid
flowchart LR
    A[🎬 Video] --> B[1 · Frame extraction]
    B --> C[2 · Person tracking<br/>pose · posture · pointing]
    C --> D[3 · Gaze estimation]
    B --> E[4 · Object detection]
    D --> F[5 · Gaze–object association]
    E --> F
    F --> G[📊 Behavior labels]
```

| Stage | Model and setting |
| --- | --- |
| Person detection | Detectron2 Keypoint R-CNN R50-FPN 3x (COCO), score ≥ 0.5, first frame only |
| Tracking | TransT, one tracker per person |
| Pose | MMPose RTMPose-m (SimCC), 256×192, 17 COCO keypoints |
| Gaze | Gaze-LLE `gazelle_dinov2_vitl14_inout`; head box from keypoints 0–4 |
| Objects | MMDetection RTMDet-L (COCO), laptop class, score ≥ 0.3 |
| Speech (not in this code) | pyannote, Whisper, TalkNet |

## 🖼️ What the pipeline sees

<table>
  <tr>
    <td colspan="3" align="center"><img src="figures/fig0.2_person_detection.png" alt="Person detection"><br><sub><b>Person detection:</b> three individuals, each segmented</sub></td>
  </tr>
  <tr>
    <td align="center" width="33%"><img src="figures/fig0.3a_gaze_scenario.jpg" alt="Gaze scenario A"></td>
    <td align="center" width="33%"><img src="figures/fig0.3b_gaze_scenario.jpg" alt="Gaze scenario B"></td>
    <td align="center" width="33%"><img src="figures/fig0.3c_gaze_scenario.jpg" alt="Gaze scenario C"></td>
  </tr>
  <tr><td colspan="3" align="center"><sub><b>Gaze estimation</b> in three scenarios</sub></td></tr>
  <tr>
    <td align="center"><img src="figures/fig0.4a_no_pointing.png" alt="No pointing"><br><sub>(a) No pointing</sub></td>
    <td align="center"><img src="figures/fig0.4b_pointing_detected.png" alt="Pointing detected"><br><sub>(b) Pointing detected</sub></td>
    <td align="center"><img src="figures/fig0.5_laptop_rotation.png" alt="Laptop rotation"><br><sub>Laptop rotated toward the group</sub></td>
  </tr>
</table>

## 📊 Results

### Against expert NICE annotations

<img src="figures/charts/nice_results.svg" alt="NICE per-behavior accuracy and F1" width="100%">

Movement-based behaviors (looking away, pointing, leaning away) are recognized reliably; gaze *targets* (joint and individual gaze) are the hardest. Setup and error analysis: [`results/nice/`](results/nice/).

### Against Tobii eye tracking

<img src="figures/charts/tobii_sessions.svg" alt="Mean F1 per session" width="100%">

<img src="figures/charts/tobii_summary.svg" alt="Pooled sensitivity, specificity and balanced accuracy" width="100%">

Head direction almost always says "screen": people shift their eyes while their head keeps facing forward. Details: [`results/tobii/`](results/tobii/).

<details>
<summary>Regenerate the charts</summary>

```bash
python results/make_charts.py               # writes figures/charts/*.svg (stdlib only)
python results/tobii/summarize.py individual_attention
```
</details>

## 🚀 Quick start

```bash
cd code
pip install -r requirements.txt --no-binary mmcv --no-build-isolation
./scripts/download_weights.sh
python server.py
```

Weights and third-party repos (TransT, Detectron2, Gaze-LLE, MMDetection) are not included. See [`code/README.md`](code/README.md) for Docker and GPU setup.

## 📁 Repository layout

| Folder | What's inside |
| --- | --- |
| [`code/`](code/) | The five-stage pipeline (`multimodal-attention-analyzer`) |
| [`figures/`](figures/) | Chapter figures 0.1–0.5 at full resolution, plus generated result charts |
| [`results/nice/`](results/nice/) | Per-behavior accuracy and F1 vs. expert NICE annotations |
| [`results/tobii/`](results/tobii/) | Head-based gaze vs. Tobii eye tracking, 12 sessions |
| [`papers/`](papers/) | The chapter and the AAAI 2025 paper, plus [`PAPERS.md`](papers/PAPERS.md) |

## 📚 Papers

- **Zhu, Y., Dey, I., Lai, K., & Pustejovsky, J. (to appear).** *Nonverbal Behavior Recognition in Multimodal Interactions from RGB Video.* Book chapter. [PDF](papers/Zhu_Dey_Lai_Pustejovsky_Nonverbal_Behavior_Recognition_RGB_Video_chapter.pdf)
- **Palmer, D., Zhu, Y., et al. (2025).** *Speech Is Not Enough: Interpreting Nonverbal Behavior in Multiparty Dialogue.* AAAI 2025. [PDF](papers/Palmer_Zhu_et_al_2025_Speech_Is_Not_Enough_AAAI.pdf)

More related work in [`papers/PAPERS.md`](papers/PAPERS.md).

## 🔒 Privacy

The figures come from the chapter as prepared for publication. Raw study recordings are IRB-protected and are not included. Check with the protocol before sharing anything here outside the lab.
