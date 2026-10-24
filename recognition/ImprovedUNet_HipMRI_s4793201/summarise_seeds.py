"""
summarise_seeds.py - aggregate test metrics over repeated training runs (different seeds).

Each run of train.py writes runs/<name>/test_metrics.json. This script groups runs by model
(the "model" field in the json), and prints mean ± sample standard deviation for the key
metrics as a GitHub-markdown table. It also saves figures/seed_summary.json.

Example (run from this folder):
    python summarise_seeds.py --runs runs/improved runs/improved_s1 runs/improved_s2 \
                                     runs/unet runs/unet_s1 runs/unet_s2
"""
import argparse
import json
import math
import os
import statistics as st

PRETTY = {"improved": "Improved U-Net", "unet": "U-Net (baseline)"}
CLASSES = ["body", "bone", "bladder", "rectum", "prostate"]


def load(path):
    with open(os.path.join(path, "test_metrics.json")) as f:
        return json.load(f)


def flat(m):
    """Pull the comparable scalar metrics out of one test_metrics.json."""
    out = {f"Dice {c}": m["pooled_dice"][c] for c in CLASSES}
    out["IoU prostate"] = m["pooled_iou"]["prostate"]
    out["Prostate Dice, base/apex slices"] = m["prostate_dice_boundary"]
    out["Prostate Dice, mid-gland slices"] = m["prostate_dice_mid"]
    out["Boundary gap (mid - base/apex)"] = m["prostate_dice_mid"] - m["prostate_dice_boundary"]
    out["False-alarm rate (prostate-free slices)"] = m["false_alarm_rate"]
    out["Worst-patient prostate Dice"] = min(m["per_patient_prostate_dice"].values())
    out["Best epoch"] = m["best_epoch"]
    out["Training time (min)"] = m["train_time_min"]
    out["Latency (ms/slice)"] = m["latency_ms_per_slice"]
    out["Training peak VRAM (MB)"] = m["train_peak_vram_mb"]
    out["Parameters (M)"] = m["parameters"] / 1e6
    return out


def fmt(vals, digits):
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{st.mean(vals):.{digits}f} ± {st.stdev(vals):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run folders containing test_metrics.json")
    ap.add_argument("--out", default="figures/seed_summary.json")
    args = ap.parse_args()

    groups = {}
    for r in args.runs:
        if not os.path.exists(os.path.join(r, "test_metrics.json")):
            print(f"skip {r}: no test_metrics.json (run not finished?)")
            continue
        m = load(r)
        groups.setdefault(m["model"], []).append(flat(m))

    models = [k for k in ("unet", "improved") if k in groups]
    keys = list(groups[models[0]][0])
    print(f"runs per model: { {PRETTY[k]: len(groups[k]) for k in models} }\n")
    print("| Test metric (mean ± std over seeds) | " + " | ".join(PRETTY[k] for k in models) + " |")
    print("| --- |" + " --- |" * len(models))
    summary = {}
    for key in keys:
        digits = 1 if key in ("Best epoch", "Training time (min)", "Training peak VRAM (MB)") else \
                 2 if key in ("Latency (ms/slice)", "Parameters (M)") else 3
        row = []
        for k in models:
            vals = [g[key] for g in groups[k] if math.isfinite(g[key])]  # NaN e.g. VRAM on CPU
            summary.setdefault(PRETTY[k], {})[key] = {"mean": st.mean(vals) if vals else None,
                                                      "std": st.stdev(vals) if len(vals) > 1 else 0.0,
                                                      "runs": vals}
            row.append(fmt(vals, digits))
        print(f"| {key} | " + " | ".join(row) + " |")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
