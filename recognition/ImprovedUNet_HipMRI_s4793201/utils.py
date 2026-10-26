"""
utils.py - metrics, evaluation loop, profiling and plotting helpers shared by train.py
and predict.py.

Metric definitions (per class c, computed from pixel counts):
    Dice(c) = 2|P ∩ G| / (|P| + |G|)        IoU(c) = |P ∩ G| / |P ∪ G|
"Pooled" metrics sum the counts over all slices of the split before dividing (the
standard volumetric-style test score). "Per-slice" metrics divide on every slice first
and are only defined where the class is present in the ground truth.
"""
import json
import os
import random
import time

import numpy as np
import torch

from dataset import CLASS_NAMES, NUM_CLASSES, PROSTATE


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- metrics
def confusion_counts(pred, target, num_classes=NUM_CLASSES):
    """Per-sample, per-class (intersection, |pred|, |gt|) counts.

    pred/target: [B,H,W] (slices) or [B,X,Y,Z] (volumes) long. Returns three tensors [B, C].
    """
    p = torch.nn.functional.one_hot(pred, num_classes).flatten(1, -2)    # [B, voxels, C]
    g = torch.nn.functional.one_hot(target, num_classes).flatten(1, -2)
    inter = (p & g).sum(1)
    return inter, p.sum(1), g.sum(1)


def dice_iou_from_counts(inter, psum, gsum):
    """Dice and IoU from (summed) counts; NaN where both prediction and GT are empty."""
    inter, psum, gsum = (np.asarray(a, dtype=np.float64) for a in (inter, psum, gsum))
    denom = psum + gsum
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = np.where(denom > 0, 2 * inter / denom, np.nan)
        iou = np.where(denom - inter > 0, inter / (denom - inter), np.nan)
    return dice, iou


@torch.no_grad()
def predict_split(model, loader, device, amp=True):
    """Run the model over a loader and collect per-slice counts, predictions and keys.

    Returns dict with numpy arrays: inter/psum/gsum [N,C], keys [N,3], preds [N,H,W] uint8.
    """
    model.eval()
    inters, psums, gsums, keys, preds = [], [], [], [], []
    for x, y, k in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        pred = logits.argmax(1)
        i, p, g = confusion_counts(pred, y)
        inters.append(i.cpu()); psums.append(p.cpu()); gsums.append(g.cpu())
        keys.append(k); preds.append(pred.to(torch.uint8).cpu())
    cat = lambda lst: torch.cat(lst).numpy()
    return {"inter": cat(inters), "psum": cat(psums), "gsum": cat(gsums),
            "keys": cat(keys), "preds": cat(preds)}


def summarise(res, positions=None, fp_threshold=50):
    """Turn per-slice counts into the report's metric dictionary.

    positions: optional array of 'boundary'|'mid'|'none' per slice (dataset.gland_position).
    fp_threshold: prostate pixels predicted on a prostate-free slice that count as a false alarm.
    """
    inter, psum, gsum, keys = res["inter"], res["psum"], res["gsum"], res["keys"]
    dice, iou = dice_iou_from_counts(inter.sum(0), psum.sum(0), gsum.sum(0))
    out = {"pooled_dice": dict(zip(CLASS_NAMES, dice.round(4).tolist())),
           "pooled_iou": dict(zip(CLASS_NAMES, iou.round(4).tolist()))}

    # per-slice prostate Dice on slices that contain prostate
    sd, _ = dice_iou_from_counts(inter[:, PROSTATE], psum[:, PROSTATE], gsum[:, PROSTATE])
    has_p = gsum[:, PROSTATE] > 0
    out["prostate_slice_dice_mean"] = float(np.nanmean(sd[has_p])) if has_p.any() else float("nan")

    # false alarms on prostate-free slices
    empty = ~has_p
    fp = psum[empty, PROSTATE] > fp_threshold
    out["prostate_free_slices"] = int(empty.sum())
    out["false_alarm_rate"] = float(fp.mean()) if empty.any() else float("nan")

    # per-patient pooled prostate Dice
    per_pat = {}
    for p in np.unique(keys[:, 0]):
        m = keys[:, 0] == p
        d, _ = dice_iou_from_counts(inter[m, PROSTATE].sum(), psum[m, PROSTATE].sum(), gsum[m, PROSTATE].sum())
        per_pat[int(p)] = round(float(d), 4)
    out["per_patient_prostate_dice"] = per_pat

    # boundary (base/apex) vs mid-gland
    if positions is not None:
        for name in ("boundary", "mid"):
            m = positions == name
            d, _ = dice_iou_from_counts(inter[m, PROSTATE].sum(), psum[m, PROSTATE].sum(), gsum[m, PROSTATE].sum())
            out[f"prostate_dice_{name}"] = round(float(d), 4)
            out[f"n_{name}_slices"] = int(m.sum())
            out[f"prostate_slice_dice_{name}"] = float(np.nanmean(sd[m])) if m.any() else float("nan")
    return out


def per_slice_prostate_dice(res):
    d, _ = dice_iou_from_counts(res["inter"][:, PROSTATE], res["psum"][:, PROSTATE], res["gsum"][:, PROSTATE])
    return d


# --------------------------------------------------------------------------- profiling
@torch.no_grad()
def profile_inference(model, device, shape=(1, 1, 256, 128), warmup=10, reps=100):
    """Mean single-slice inference latency in ms (batch 1) and peak VRAM in MB."""
    model.eval()
    x = torch.randn(*shape, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / reps * 1000
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
    return {"latency_ms_per_slice": round(ms, 2), "inference_peak_vram_mb": round(vram, 1)}


# --------------------------------------------------------------------------- io / plots
def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_history(history, path, title=""):
    """Plot train/val loss and val Dice (mean foreground + prostate) per epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set(xlabel="epoch", ylabel="Dice + CE loss", title="Loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_dice_prostate"] for h in history], label="prostate")
    ax[1].plot(ep, [h["val_dice_mean_fg"] for h in history], label="mean of 5 foreground classes")
    ax[1].axhline(0.75, ls="--", c="grey", lw=1, label="target 0.75")
    ax[1].set(xlabel="epoch", ylabel="pooled Dice", title="Validation Dice", ylim=(0, 1))
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
