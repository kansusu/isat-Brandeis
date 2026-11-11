"""Render the README result charts as SVG (standard library only).

Usage: python results/make_charts.py
Writes figures/charts/*.svg. Each SVG follows the viewer's light/dark setting.
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "charts")
os.makedirs(OUT, exist_ok=True)

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
STYLE = f"""<style>
  svg {{ --ink:#1f2328; --ink2:#59636e; --grid:#d1d9e0; --s1:#2a78d6; --s2:#eb6834; --bad:#e34948; }}
  @media (prefers-color-scheme: dark) {{
    svg {{ --ink:#f0f6fc; --ink2:#9198a1; --grid:#3d444d; --s1:#3987e5; --s2:#d95926; --bad:#e66767; }}
  }}
  text {{ font-family: {FONT}; fill: var(--ink); }}
  .t {{ font-size:18px; font-weight:600; }}
  .st {{ font-size:13px; fill: var(--ink2); }}
  .lab {{ font-size:13px; }}
  .ax {{ font-size:11px; fill: var(--ink2); }}
  .val {{ font-size:12px; font-weight:600; }}
  .grid {{ stroke: var(--grid); stroke-width:1; }}
  .s1 {{ fill: var(--s1); }} .s2 {{ fill: var(--s2); }}
</style>"""


def bar_h(x, y, w, h, cls, r=4):
    """Horizontal bar with rounded data end (right), square at baseline."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f'<path class="{cls}" d="M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} '
            f'V{y + h - r} Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z"/>')


def bar_v(x, y0, w, h, cls, r=4):
    """Vertical bar rising from baseline y0, rounded top."""
    if h <= 0:
        return ""
    r = min(r, h, w / 2)
    top = y0 - h
    return (f'<path class="{cls}" d="M{x},{y0} V{top + r} Q{x},{top} {x + r},{top} '
            f'H{x + w - r} Q{x + w},{top} {x + w},{top + r} V{y0} Z"/>')


def legend(x, y, items):
    out, cx = [], x
    for cls, label in items:
        out.append(f'<rect class="{cls}" x="{cx}" y="{y - 10}" width="12" height="12" rx="3"/>')
        out.append(f'<text class="lab" x="{cx + 18}" y="{y}">{label}</text>')
        cx += 30 + len(label) * 7.2
    return "".join(out)


def write(name, w, h, body, title):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'role="img" aria-label="{title}">{STYLE}{body}</svg>\n')
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)
    print("wrote", os.path.relpath(os.path.join(OUT, name)))


# ---------- 1. NICE per-behavior accuracy and F1 ----------
def nice_chart():
    rows = list(csv.DictReader(open(os.path.join(HERE, "nice", "nice_results.csv"))))
    overall = next(r for r in rows if r["behavior"] == "overall")
    rows = sorted((r for r in rows if r["behavior"] != "overall"), key=lambda r: -float(r["f1"]))
    W, left, right, top = 760, 150, 60, 96
    pw = W - left - right
    bh, gap, group = 11, 2, 38
    H = top + group * len(rows) + 40
    b = [f'<text class="t" x="0" y="22">NICE evaluation: per-behavior accuracy and F1</text>',
         f'<text class="st" x="0" y="44">12 clips vs. expert annotations · overall accuracy '
         f'{float(overall["accuracy"]):.2f}, F1 {float(overall["f1"]):.2f}</text>',
         legend(0, 74, [("s1", "Accuracy"), ("s2", "F1")])]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + pw * t
        b.append(f'<line class="grid" x1="{x}" x2="{x}" y1="{top - 8}" y2="{H - 34}"/>')
        b.append(f'<text class="ax" x="{x}" y="{H - 18}" text-anchor="middle">{t:.2f}</text>')
    for i, r in enumerate(rows):
        y = top + i * group
        name = r["behavior"].replace("_", " ").capitalize()
        acc, f1 = float(r["accuracy"]), float(r["f1"])
        b.append(f'<text class="lab" x="{left - 12}" y="{y + bh + 5}" text-anchor="end">{name}</text>')
        b.append(bar_h(left, y, pw * acc, bh, "s1"))
        b.append(bar_h(left, y + bh + gap, pw * f1, bh, "s2"))
        b.append(f'<text class="ax" x="{left + pw * acc + 6}" y="{y + 9}">{acc:.2f}</text>')
        b.append(f'<text class="val" x="{left + pw * f1 + 6}" y="{y + bh + gap + 10}">{f1:.2f}</text>')
    write("nice_results.svg", W, H, "".join(b), "NICE per-behavior accuracy and F1")


# ---------- 2. Tobii mean F1 by session ----------
def tobii_sessions():
    def load(folder):
        p = os.path.join(HERE, "tobii", folder, f"{folder}_mean_f1_by_session.csv")
        return {r["session"]: float(r["mean_f1_percent"]) for r in csv.DictReader(open(p))}
    ind, grp = load("individual_attention"), load("group_evaluation")
    sessions = sorted(ind)
    W, H, left, top, bottom = 760, 380, 44, 96, 330
    pw = W - left - 10
    slot = pw / len(sessions)
    bw = 16
    b = ['<text class="t" x="0" y="22">Head-based gaze vs. Tobii eye tracking: mean F1 per session</text>',
         f'<text class="st" x="0" y="44">12 sessions · mean of session means: individual '
         f'{sum(ind.values()) / len(ind):.1f}%, group {sum(grp.values()) / len(grp):.1f}%</text>',
         legend(0, 74, [("s1", "Individual attention"), ("s2", "Group attention")])]
    ph = bottom - top
    for t in (0, 25, 50, 75, 100):
        y = bottom - ph * t / 100
        b.append(f'<line class="grid" x1="{left}" x2="{W}" y1="{y}" y2="{y}"/>')
        b.append(f'<text class="ax" x="{left - 8}" y="{y + 4}" text-anchor="end">{t}%</text>')
    for i, s in enumerate(sessions):
        cx = left + slot * i + slot / 2
        b.append(bar_v(cx - bw - 1, bottom, bw, ph * ind[s] / 100, "s1"))
        b.append(bar_v(cx + 1, bottom, bw, ph * grp[s] / 100, "s2"))
        b.append(f'<text class="ax" x="{cx}" y="{bottom + 18}" text-anchor="middle">{s}</text>')
    b.append(f'<text class="ax" x="{left + pw / 2}" y="{bottom + 40}" text-anchor="middle">Session</text>')
    write("tobii_sessions.svg", W, H, "".join(b), "Mean F1 per session, individual and group attention")


# ---------- 3. Tobii pooled sensitivity / specificity ----------
def tobii_tiles():
    tp = fp = tn = fn = 0.0
    for p in glob.glob(os.path.join(HERE, "tobii", "individual_attention", "*_report_session_*.csv")):
        for r in csv.DictReader(open(p)):
            tp += float(r["TP"]); fp += float(r["FP"]); tn += float(r["TN"]); fn += float(r["FN"])
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    tiles = [("Sensitivity", sens, "looking at screen, caught"),
             ("Specificity", spec, "looking away, caught"),
             ("Balanced accuracy", (sens + spec) / 2, "≈ chance (0.5)")]
    W, H, tw = 760, 168, 760 / 3
    b = []
    for i, (name, v, note) in enumerate(tiles):
        x = i * tw
        b.append(f'<text class="st" x="{x}" y="20">{name}</text>')
        b.append(f'<text x="{x}" y="72" style="font-size:44px;font-weight:600">{v:.3f}</text>')
        # meter
        mw = tw - 40
        b.append(f'<rect x="{x}" y="90" width="{mw}" height="8" rx="4" style="fill:var(--grid)"/>')
        b.append(f'<rect class="s1" x="{x}" y="90" width="{mw * v}" height="8" rx="4"/>')
        b.append(f'<text class="ax" x="{x}" y="122">{note}</text>')
    b.append(f'<text class="ax" x="0" y="{H - 6}">Pooled frames, 12 sessions: TP {tp:,.0f} · FP {fp:,.0f} · '
             f'TN {tn:,.0f} · FN {fn:,.0f}</text>')
    write("tobii_summary.svg", W, H, "".join(b), "Pooled sensitivity, specificity and balanced accuracy")


if __name__ == "__main__":
    nice_chart()
    tobii_sessions()
    tobii_tiles()
