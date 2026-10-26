"""
predict.py - example usage of trained models: inference on the held-out test split,
metric comparison and failure-case visualisations.

Examples (run from this folder after train.py):
    # evaluate the proposed model and compare it against the baseline
    python predict.py --data-root <keras_slices_data> \
        --ckpt runs/improved/best.pt --baseline runs/unet/best.pt

    # just one model
    python predict.py --data-root <keras_slices_data> --ckpt runs/improved/best.pt

Outputs (in --out, default figures/):
    examples.png           input | ground truth | prediction for typical test slices
    failures_<model>.png   the k worst prostate slices: input, GT, prediction, error overlay
                           (red = false positive, blue = false negative) for the autopsy
    curves_<model>.png     training curves copied from runs/<model>/
    size_dice.png          prostate Dice binned by the slice's true prostate area (px)
    position_dice.png      prostate Dice on base/apex (boundary) vs mid-gland slices
    comparison.json        side-by-side metrics for README tables
"""
import argparse
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, PROSTATE, HipMRISlices, gland_position
from modules import build_model, count_parameters
from utils import (get_device, per_slice_prostate_dice, predict_split, profile_inference,
                   save_json, summarise)

LABEL_CMAP = ListedColormap(["#000000", "#5a5a5a", "#e8e8e8", "#f2c14e", "#3fa7d6", "#e4572e"])


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device)
    model = build_model(ck["model"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return ck["model"], model


def evaluate(name, model, loader, ds, device, positions):
    res = predict_split(model, loader, device)
    m = summarise(res, positions)
    m["parameters"] = count_parameters(model)
    m.update(profile_inference(model, device))
    print(f"\n=== {name}")
    for c in CLASS_NAMES[1:]:
        print(f"  {c:<9} Dice {m['pooled_dice'][c]:.4f}  IoU {m['pooled_iou'][c]:.4f}")
    print(f"  prostate: boundary {m['prostate_dice_boundary']:.4f} ({m['n_boundary_slices']} slices) | "
          f"mid {m['prostate_dice_mid']:.4f} ({m['n_mid_slices']} slices)")
    print(f"  false alarms on {m['prostate_free_slices']} prostate-free slices: {m['false_alarm_rate']:.3f}")
    print(f"  per patient: {m['per_patient_prostate_dice']}")
    print(f"  params {m['parameters'] / 1e6:.2f} M | {m['latency_ms_per_slice']} ms/slice | "
          f"inference VRAM {m['inference_peak_vram_mb']} MB")
    return res, m


def _error_overlay(gt, pred):
    """RGBA overlay: green = correct prostate, red = false positive, blue = false negative.

    Pixels with no prostate in either mask are fully transparent.
    """
    g, p = gt == PROSTATE, pred == PROSTATE
    rgba = np.zeros(gt.shape + (4,))
    rgba[g & p] = (0.2, 0.8, 0.3, 0.6)
    rgba[p & ~g] = (0.95, 0.15, 0.15, 0.85)
    rgba[g & ~p] = (0.15, 0.4, 1.0, 0.85)
    return rgba


def plot_failures(ds, res, positions, name, path, k=5):
    """Worst-k prostate-containing slices by per-slice prostate Dice, plus one false alarm."""
    d = per_slice_prostate_dice(res)
    has_p = res["gsum"][:, PROSTATE] > 0
    cand = np.where(has_p)[0]
    worst = cand[np.argsort(np.nan_to_num(d[cand], nan=0))][:k - 1].tolist()
    empty = np.where(~has_p)[0]
    if len(empty):  # worst false alarm on a prostate-free slice
        worst.append(int(empty[np.argmax(res["psum"][empty, PROSTATE])]))
    fig, ax = plt.subplots(len(worst), 4, figsize=(9, 3.1 * len(worst)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(worst):
        p, w, s = ds.keys[i]
        img, gt, pr = ds.images[i].astype(np.float32), ds.masks[i], res["preds"][i]
        ax[r, 0].imshow(img, cmap="gray")
        ax[r, 1].imshow(gt, cmap=LABEL_CMAP, vmin=0, vmax=5, interpolation="nearest")
        ax[r, 2].imshow(pr, cmap=LABEL_CMAP, vmin=0, vmax=5, interpolation="nearest")
        ax[r, 3].imshow(img, cmap="gray")
        ax[r, 3].imshow(_error_overlay(gt, pr), interpolation="nearest")
        dice = "no GT prostate" if not has_p[i] else f"prostate Dice {d[i]:.2f}"
        ax[r, 0].set_ylabel(f"pt {p} wk {w} sl {s}\n{positions[i]}\n{dice}", fontsize=8)
    for a, t in zip(ax[0], ["input", "ground truth", "prediction", "prostate: green TP, red FP, blue FN"]):
        a.set_title(t, fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{name}: worst test slices", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [int(i) for i in worst]


def plot_examples(ds, preds_by_model, path, n=4, seed=0):
    """Typical mid-gland test slices: input, GT and each model's prediction."""
    rng = np.random.default_rng(seed)
    area = (ds.masks == PROSTATE).reshape(len(ds), -1).sum(1)
    pool = np.where(area > 300)[0]  # typical, clearly visible mid-gland slices
    if len(pool) < n:  # small glands: fall back to the largest prostate slices
        pool = np.argsort(area)[::-1][:max(n, 1)]
    idx = np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))
    cols = 2 + len(preds_by_model)
    fig, ax = plt.subplots(len(idx), cols, figsize=(2.4 * cols, 3.2 * len(idx)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(idx):
        ax[r, 0].imshow(ds.images[i].astype(np.float32), cmap="gray")
        ax[r, 1].imshow(ds.masks[i], cmap=LABEL_CMAP, vmin=0, vmax=5, interpolation="nearest")
        for c, (name, pr) in enumerate(preds_by_model.items()):
            ax[r, 2 + c].imshow(pr[i], cmap=LABEL_CMAP, vmin=0, vmax=5, interpolation="nearest")
    for a, t in zip(ax[0], ["input", "ground truth"] + list(preds_by_model)):
        a.set_title(t, fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_position(metrics_by_model, path):
    names = list(metrics_by_model)
    groups = ["boundary", "mid"]
    x = np.arange(len(groups))
    wdt = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for j, n in enumerate(names):
        vals = [metrics_by_model[n][f"prostate_dice_{g}"] for g in groups]
        bars = ax.bar(x + j * wdt - 0.4 + wdt / 2, vals, wdt, label=n)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, ["base / apex slices", "mid-gland slices"])
    ax.axhline(0.75, ls="--", c="grey", lw=1)
    ax.set(ylabel="pooled prostate Dice (test)", ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


SIZE_BINS = [(1, 50), (50, 150), (150, 400), (400, 10 ** 6)]


def size_breakdown(ds, res):
    """Pooled prostate Dice and miss count, binned by the slice's ground-truth prostate area (px)."""
    area = (ds.masks == PROSTATE).reshape(len(ds), -1).sum(1)
    d = per_slice_prostate_dice(res)
    rows = []
    for lo, hi in SIZE_BINS:
        m = (area >= lo) & (area < hi)
        inter, ps, gs = (res[k][m, PROSTATE].sum() for k in ("inter", "psum", "gsum"))
        rows.append({"bin": f"{lo}-{hi - 1}" if hi < 10 ** 6 else f">={lo}", "slices": int(m.sum()),
                     "pooled_dice": round(float(2 * inter / max(ps + gs, 1)), 4),
                     "missed_slices": int((d[m] < 0.1).sum())})
    return rows


def error_breakdown(ds, res):
    """Where prostate errors go: true label under false-positive prostate pixels, and the label
    predicted on missed prostate pixels. Leakage into rectum (4) or bladder (3) is the clinically
    hazardous case (over-irradiating an organ at risk)."""
    gt, pr = ds.masks, res["preds"]
    fp = gt[(pr == PROSTATE) & (gt != PROSTATE)]
    fn = pr[(gt == PROSTATE) & (pr != PROSTATE)]
    count = lambda a: {CLASS_NAMES[int(k)]: int(v) for k, v in zip(*np.unique(a, return_counts=True))}
    return {"false_positive_pixels_on": count(fp), "missed_pixels_predicted_as": count(fn)}


def extent_breakdown(ds, res):
    """Per scan, compare the slice range that contains prostate in the ground truth vs prediction.

    Returns one row per (patient, week): true and predicted first/last prostate slice and the
    signed end errors in slices. start_err > 0 / end_err < 0 = gland ends missed (under-contour);
    start_err < 0 / end_err > 0 = predicted gland runs past the true end (over-contour into OARs).
    """
    rows = []
    for p, w in sorted({(int(a), int(b)) for a, b, _ in ds.keys}):
        m = (ds.keys[:, 0] == p) & (ds.keys[:, 1] == w)
        sl = ds.keys[m, 2]

        def span(labels):
            s = sl[(labels[m] == PROSTATE).reshape(m.sum(), -1).any(1)]
            return (int(s.min()), int(s.max())) if len(s) else (None, None)

        g, q = span(ds.masks), span(res["preds"])
        row = {"patient": p, "week": w, "true": g, "pred": q}
        if None not in g + q:
            row.update(start_err=q[0] - g[0], end_err=q[1] - g[1])
        rows.append(row)
    return rows


def plot_size(rows_by_model, path):
    names = list(rows_by_model)
    bins = [r["bin"] for r in rows_by_model[names[0]]]
    x = np.arange(len(bins))
    wdt = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for j, n in enumerate(names):
        bars = ax.bar(x + j * wdt - 0.4 + wdt / 2, [r["pooled_dice"] for r in rows_by_model[n]], wdt, label=n)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    counts = [r["slices"] for r in rows_by_model[names[0]]]
    ax.set_xticks(x, [f"{b} px\n({c} slices)" for b, c in zip(bins, counts)], fontsize=8)
    ax.axhline(0.75, ls="--", c="grey", lw=1)
    ax.set(xlabel="ground-truth prostate area on the slice", ylabel="pooled prostate Dice (test)", ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--ckpt", required=True, help="trained proposed model (best.pt)")
    ap.add_argument("--baseline", default=None, help="trained baseline (best.pt) to compare against")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = get_device()

    ds = HipMRISlices(args.data_root, "test", args.cache_dir)
    loader = DataLoader(ds, 32, shuffle=False)
    positions = gland_position(ds.keys, ds.masks)

    pretty = {"improved": "Improved U-Net", "unet": "U-Net (baseline)"}
    results, metrics = {}, {}
    for path in [p for p in (args.baseline, args.ckpt) if p]:
        kind, model = load_ckpt(path, device)
        name = pretty[kind]
        results[name], metrics[name] = evaluate(name, model, loader, ds, device, positions)
        curves = os.path.join(os.path.dirname(path), "curves.png")  # written by train.py
        if os.path.exists(curves):
            shutil.copy(curves, os.path.join(args.out, f"curves_{kind}.png"))
        metrics[name]["worst_slices"] = plot_failures(
            ds, results[name], positions, name, os.path.join(args.out, f"failures_{kind}.png"))
        metrics[name]["dice_by_prostate_area"] = size_breakdown(ds, results[name])
        metrics[name]["prostate_errors"] = error_breakdown(ds, results[name])
        ext = metrics[name]["gland_extent"] = extent_breakdown(ds, results[name])
        over = sum(r.get("start_err", 0) < 0 or r.get("end_err", 0) > 0 for r in ext)
        print(f"  gland extent (true vs predicted slice range): over-runs the true end in {over}/{len(ext)} scans")
        for r in ext:
            print(f"    pt {r['patient']} wk {r['week']}: true {r['true']} pred {r['pred']}")
        print(f"  Dice by prostate area: {metrics[name]['dice_by_prostate_area']}")
        print(f"  prostate errors: {metrics[name]['prostate_errors']}")

    plot_examples(ds, {n: r["preds"] for n, r in results.items()}, os.path.join(args.out, "examples.png"))
    plot_position(metrics, os.path.join(args.out, "position_dice.png"))
    plot_size({n: m["dice_by_prostate_area"] for n, m in metrics.items()}, os.path.join(args.out, "size_dice.png"))
    save_json(metrics, os.path.join(args.out, "comparison.json"))

    if len(metrics) == 2:
        a, b = list(metrics)
        print("\n| metric | " + a + " | " + b + " |\n| --- | --- | --- |")
        for c in CLASS_NAMES[1:]:
            print(f"| Dice {c} | {metrics[a]['pooled_dice'][c]:.3f} | {metrics[b]['pooled_dice'][c]:.3f} |")
        for k in ("prostate_dice_boundary", "prostate_dice_mid", "false_alarm_rate",
                  "latency_ms_per_slice", "parameters"):
            print(f"| {k} | {metrics[a][k]} | {metrics[b][k]} |")
    print(f"\nfigures written to {args.out}/")


if __name__ == "__main__":
    main()
