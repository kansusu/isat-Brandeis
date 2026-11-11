# Evaluation against NICE annotations

## Setup

- Jigsaw collaborative task, 2 groups × 3 students
- 12 clips, 10–25 s each, 60 FPS
- Gold labels: expert NICE annotations per utterance (79.6% / 79.5% agreement with two other annotators)
- Frame-level predictions aggregated per utterance: a behavior counts as present if it appears in at least one frame inside the utterance window

## Results

| Behavior | Accuracy | F1 |
| --- | --- | --- |
| Look away | 0.96 | 0.82 |
| Point at others | 0.93 | 0.77 |
| Lean away | 0.94 | 0.77 |
| Point at tool | 0.89 | 0.74 |
| Lean towards | 0.62 | 0.46 |
| Look at others | 0.64 | 0.45 |
| Joint gaze | 0.46 | 0.37 |
| Individual gaze | 0.46 | 0.32 |
| **Overall** | **0.73** | **0.59** |

The same numbers are in [`nice_results.csv`](nice_results.csv).

## Error analysis

- **Same task, different objects:** when one student types and another watches the screen, annotators code joint attention on the shared task; the model sees two targets. Counting keyboard + screen as one target resolves every such case.
- **Facing ≠ looking:** see [`../tobii/`](../tobii/).
- **Frame vs. behavior:** the model decides every 1/60 s; annotators code about 2 s of sustained change.
- **One view vs. three:** the model sees one camera with the screen off-camera; annotators had three synchronized views.

Pilot-scale evaluation. Source: Zhu, Dey, Lai & Pustejovsky (to appear), *Nonverbal Behavior Recognition in Multimodal Interactions from RGB Video.*
