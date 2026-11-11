"""Pool the per-session Tobii comparison reports into sensitivity, specificity and balanced accuracy.

Usage: python summarize.py individual_attention
"""
import csv
import glob
import os
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "individual_attention"
here = os.path.dirname(os.path.abspath(__file__))
tp = fp = tn = fn = 0
for path in sorted(glob.glob(os.path.join(here, folder, "*_report_session_*.csv"))):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tp += float(row["TP"])
            fp += float(row["FP"])
            tn += float(row["TN"])
            fn += float(row["FN"])

sens = tp / (tp + fn)
spec = tn / (tn + fp)
print(f"TP {tp:,.0f}  FP {fp:,.0f}  TN {tn:,.0f}  FN {fn:,.0f}")
print(f"sensitivity {sens:.3f}  specificity {spec:.3f}  balanced accuracy {(sens + spec) / 2:.3f}")
