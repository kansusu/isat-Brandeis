# Head-based gaze vs. Tobii eye tracking

Frame-level comparison between the pipeline's head-based gaze estimate (Gaze-LLE) and Tobii eye tracking, over 12 sessions (316–352). The screen is a fixed region, so object detection is not involved.

## Pooled result (individual attention)

Positive class = looking at the screen.

| Metric | Value |
| --- | --- |
| Sensitivity (looking at screen caught) | 0.911 |
| Specificity (looking away caught) | 0.118 |
| Balanced accuracy | 0.515 |

Counts summed over all sessions, clips and people: TP 582,312 · FP 293,612 · TN 39,195 · FN 56,611. Recompute with:

```bash
python summarize.py individual_attention
```

The estimate almost always says "screen": small gaze shifts happen with the eyes while the head keeps facing the screen.

## Files

- `individual_attention/individual_attention_report_session_<id>.csv`: one row per clip × person, with `Matched_Color` (the person's overlay color), F1, accuracy, precision, recall, and TP / FP / TN / FN frame counts.
- `individual_attention/individual_attention_mean_f1_by_session.csv`: mean F1 per session.
- `group_evaluation/group_evaluation_report_session_<id>.csv`: one row per clip, with the percentage of frames predicted vs. gold, TP / FP / FN / TN (in percentage points), accuracy and F1.
- `group_evaluation/group_evaluation_mean_f1_by_session.csv`: mean F1 per session.

Aggregate scores only; no video, frames or eye-tracking streams are included.
