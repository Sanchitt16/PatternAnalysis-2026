"""
predict3d.py - evaluate the 3D Improved U-Net and compare it with the 2D models on *identical
pixels* of the held-out test patients (40-42).

How the comparison is made fair. The provided 2D slices are not a separate dataset: slice k of
a scan is exactly volume[:, 98 + k, :] flipped along the first axis (checked pixel-for-pixel
for every test scan below, and the script stops if it does not hold). So the 3D model
  1. predicts on the downsampled volume (128 x 128 x 64),
  2. has its class probabilities trilinearly upsampled back to the full 256 x 256 x 128 grid,
  3. is then read out on exactly the 60 slices per scan that the 2D models were scored on.
Every metric below (pooled Dice/IoU, base/apex vs mid-gland, Dice by prostate area, error
type, gland extent, false alarms) is therefore computed on the same 540 test slices for all
three models. The 3D model's native volumetric Dice on the full grid is reported as well, and so
is the "downsampling ceiling": the score of a perfect prediction at the downsampled resolution,
i.e. what the x2 downsampling alone costs before the network makes any error.

Post-processing (post-hoc analysis, not used for model selection): every 3D checkpoint is also
scored after keeping only the largest 3D-connected prostate region ("+ largest component"),
which removes small disconnected prostate islands. Needs scipy.

Examples (run from this folder):
    python predict3d.py --study-root ../../../data \
        --ckpt3d runs/improved3d/best.pt runs/improved3d_s1/best.pt runs/improved3d_s2/best.pt \
        --ckpt2d runs/improved/best.pt --baseline2d runs/unet/best.pt

The first --ckpt3d is used for the figures; every --ckpt3d is evaluated and summarised as
mean +- std over seeds.

Outputs (in --out, default figures/):
    examples3d.png          the same test slices: input, ground truth, U-Net, Improved U-Net 2D, 3D
    failures_improved3d.png worst 3D test slices with error overlay
    autopsy_comparison.png  the README's autopsy slices, all three models side by side (zoomed)
    position_dice3d.png     prostate Dice on base/apex vs mid-gland slices, all three models
    size_dice3d.png         prostate Dice by true prostate area, all three models
    curves_improved3d.png   training curves of the first 3D run
    comparison3d.json       all metrics, per model and per 3D seed
"""
import argparse
import os
import shutil
import statistics as st

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, PROSTATE, HipMRISlices, gland_position
from dataset3d import FULL_SHAPE, HipMRIVolumes, _index, _load, fit_volume
from modules import count_parameters
from predict import (error_breakdown, evaluate, extent_breakdown, load_ckpt, plot_examples,
                     plot_failures, plot_position, plot_size, size_breakdown)
from utils import confusion_counts, get_device, profile_inference, save_json, summarise

SLICE_OFFSET = 98  # 2D slice k == volume[:, SLICE_OFFSET + k, :][::-1]


def to_2d_slices(vol, n_slices):
    """[X, Y, Z] full-resolution volume -> [n_slices, X, Z] in the 2D dataset's orientation."""
    return vol[:, SLICE_OFFSET:SLICE_OFFSET + n_slices, :].transpose(1, 0, 2)[:, ::-1, :]


def keep_largest_prostate(pred):
    """Post-processing: keep only the largest 3D-connected prostate region (26-connectivity).

    Smaller, disconnected prostate islands are relabelled as body (label 1), which is what
    surrounds them in almost every case. This is parameter-free and uses only the prediction.
    """
    from scipy import ndimage  # only needed for this optional step

    lab, n = ndimage.label(pred == PROSTATE, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return pred
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    out = pred.copy()
    out[(lab > 0) & (lab != int(np.argmax(sizes)) + 1)] = 1
    return out


@torch.no_grad()
def predict_full_res(model, x, device):
    """Downsampled input [1,1,X,Y,Z] -> full-resolution label map [256,256,128] (uint8)."""
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        logits = model(x.to(device))
    probs = logits.float().softmax(1)
    probs = F.interpolate(probs, size=FULL_SHAPE, mode="trilinear", align_corners=False)
    return probs.argmax(1)[0].to(torch.uint8).cpu().numpy()


def evaluate_3d(model, study_root, vol_ds, ds2d, device, postprocess=False):
    """Predict every test volume and read the predictions out on the 2D test slices.

    Returns (res, native): `res` has the same layout as utils.predict_split output for the 2D
    test set (so every 2D analysis function can be reused unchanged); `native` holds the full-
    resolution volumetric Dice per class.
    """
    labels = _index(os.path.join(study_root, "semantic_labels_only"))
    preds2d = np.zeros_like(ds2d.masks)
    vol_counts = np.zeros((3, 6), dtype=np.int64)
    for i in range(len(vol_ds)):
        x, _, key = vol_ds[i]
        p, w = (int(v) for v in key)
        pred = predict_full_res(model, x[None], device)
        if postprocess:
            pred = keep_largest_prostate(pred)
        gt_full = fit_volume(np.zeros(FULL_SHAPE, np.float32), np.rint(_load(labels[(p, w)])).astype(np.uint8))[1]
        # native volumetric counts on the full grid
        inter, ps, gs = (t[0].numpy() for t in confusion_counts(
            torch.from_numpy(pred.astype(np.int64))[None], torch.from_numpy(gt_full.astype(np.int64))[None]))
        vol_counts += np.stack([inter, ps, gs])
        # read out on the 2D slices of this scan
        rows = np.where((ds2d.keys[:, 0] == p) & (ds2d.keys[:, 1] == w))[0]
        rows = rows[np.argsort(ds2d.keys[rows, 2])]
        gt_slices = to_2d_slices(gt_full, len(rows))
        agree = (gt_slices == ds2d.masks[rows]).mean()
        if agree < 0.999:
            raise RuntimeError(f"scan {(p, w)}: 2D/3D alignment failed ({agree:.4f} of pixels agree)")
        preds2d[rows] = to_2d_slices(pred, len(rows))
    t = lambda a: torch.from_numpy(a.astype(np.int64))
    inter, ps, gs = confusion_counts(t(preds2d), t(ds2d.masks))
    res = {"inter": inter.numpy(), "psum": ps.numpy(), "gsum": gs.numpy(),
           "keys": ds2d.keys.astype(np.int64), "preds": preds2d}
    d = 2 * vol_counts[0] / np.maximum(vol_counts[1] + vol_counts[2], 1)
    native = {"volumetric_dice_full_res": dict(zip(CLASS_NAMES, d.round(4).tolist()))}
    return res, native


class _Oracle(torch.nn.Module):
    """'Perfect' 3D model that outputs the true downsampled labels of the volume it is given.

    Scoring it exactly like the real 3D model measures the Dice ceiling imposed by the x2
    downsampling alone (what a flawless 3D prediction at 3.1 mm loses on the full grid)."""

    def __init__(self, vol_ds):
        super().__init__()
        self.lookup = [(vol_ds[i][0], vol_ds[i][1]) for i in range(len(vol_ds))]

    def forward(self, x):
        mask = next(m for img, m in self.lookup if torch.equal(img[None], x.cpu()))
        return F.one_hot(mask, 6).movedim(-1, 0)[None].float().to(x.device) * 20


# Slices discussed in the README failure autopsy: (patient, week, slice) in 2D slice numbering.
AUTOPSY_CASES = [(40, 0, 16), (40, 0, 38), (40, 3, 14), (40, 3, 34), (41, 0, 41)]


def plot_autopsy_comparison(ds2d, results, cases, path, half=36):
    """One figure for the autopsy: rows = cases, columns = ground truth + each model.

    Each panel is the MR crop around the prostate with prostate overlaid: ground truth in
    yellow; for models green = correct, red = false positive, blue = missed. The crop is
    centred on the scan's prostate (so slices with no prostate are framed the same way).
    """
    import matplotlib.pyplot as plt

    from dataset import PROSTATE as P
    from predict import _error_overlay

    idx = {tuple(int(v) for v in k): i for i, k in enumerate(ds2d.keys)}
    names = list(results)
    fig, ax = plt.subplots(len(cases), 1 + len(names), figsize=(2.5 * (1 + len(names)), 2.6 * len(cases)))
    for r, (p, w, s) in enumerate(cases):
        i = idx[(p, w, s)]
        scan = (ds2d.keys[:, 0] == p) & (ds2d.keys[:, 1] == w)
        yy, xx = np.nonzero((ds2d.masks[scan] == P).any(0))
        cy, cx = (int(yy.mean()), int(xx.mean())) if len(yy) else (128, 64)
        y0 = int(np.clip(cy - half, 0, ds2d.masks.shape[1] - 2 * half))
        x0 = int(np.clip(cx - half, 0, ds2d.masks.shape[2] - 2 * half))
        crop = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        img, gt = ds2d.images[i].astype(np.float32)[crop], ds2d.masks[i][crop]
        gt_rgba = np.zeros(gt.shape + (4,))
        gt_rgba[gt == P] = (1.0, 0.85, 0.1, 0.7)
        ax[r, 0].imshow(img, cmap="gray")
        ax[r, 0].imshow(gt_rgba, interpolation="nearest")
        ax[r, 0].set_ylabel(f"pt {p} wk {w} sl {s}\n{int((ds2d.masks[i] == P).sum())} px true prostate", fontsize=8)
        for c, n in enumerate(names, 1):
            pr = results[n]["preds"][i]
            ax[r, c].imshow(img, cmap="gray")
            ax[r, c].imshow(_error_overlay(gt, pr[crop]), interpolation="nearest")
            g, q = ds2d.masks[i] == P, pr == P
            d = 2 * (g & q).sum() / max(g.sum() + q.sum(), 1)
            ax[r, c].text(1, 6, "no prostate (correct)" if not g.any() and not q.any() else
                          f"Dice {d:.2f}" if g.any() else f"{int(q.sum())} px false prostate",
                          color="white", fontsize=7, bbox=dict(facecolor="black", alpha=0.6, pad=1))
    for a, t in zip(ax[0], ["ground truth (yellow)"] + names):
        a.set_title(t, fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Failure cases, all models on the same slices (green correct, red false positive, blue missed)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def headline(m):
    return {"prostate Dice": m["pooled_dice"]["prostate"],
            "prostate Dice base/apex": m["prostate_dice_boundary"],
            "prostate Dice mid-gland": m["prostate_dice_mid"],
            "false-alarm rate": m["false_alarm_rate"],
            "rectum px labelled prostate": m["prostate_errors"]["false_positive_pixels_on"].get("rectum", 0),
            "scans over-running gland end": sum(r.get("start_err", 0) < 0 or r.get("end_err", 0) > 0
                                                for r in m["gland_extent"]),
            "volumetric prostate Dice (full res)": m.get("volumetric_dice_full_res", {}).get("prostate", float("nan"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", required=True,
                    help="folder with semantic_MRs, semantic_labels_only and keras_slices_data")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--ckpt3d", nargs="+", required=True)
    ap.add_argument("--ckpt2d", default=None, help="2D Improved U-Net best.pt")
    ap.add_argument("--baseline2d", default=None, help="2D U-Net best.pt")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = get_device()

    ds2d = HipMRISlices(os.path.join(args.study_root, "keras_slices_data"), "test", args.cache_dir)
    positions = gland_position(ds2d.keys, ds2d.masks)
    vol_ds = HipMRIVolumes(args.study_root, "test", args.cache_dir)

    results, metrics = {}, {}
    pretty = {"unet": "U-Net 2D (baseline)", "improved": "Improved U-Net 2D"}
    for path in [p for p in (args.baseline2d, args.ckpt2d) if p]:
        kind, model = load_ckpt(path, device)
        res, m = evaluate(pretty[kind], model, DataLoader(ds2d, 32), ds2d, device, positions)
        results[pretty[kind]], metrics[pretty[kind]] = res, m

    per_seed, per_seed_lcc = [], []
    for j, path in enumerate(args.ckpt3d):
        _, model = load_ckpt(path, device)
        # post-hoc variant: same predictions, keep only the largest connected prostate region
        res_l, native_l = evaluate_3d(model, args.study_root, vol_ds, ds2d, device, postprocess=True)
        ml = summarise(res_l, positions)
        ml.update(native_l)
        ml.update(prostate_errors=error_breakdown(ds2d, res_l), gland_extent=extent_breakdown(ds2d, res_l),
                  checkpoint=path)
        per_seed_lcc.append(ml)
        if j == 0:
            results["Improved U-Net 3D + largest component"], lcc_first = res_l, ml
        res, native = evaluate_3d(model, args.study_root, vol_ds, ds2d, device)
        m = summarise(res, positions)
        m.update(native)
        m["parameters"] = count_parameters(model)
        reps = 50 if device.type == "cuda" else 2  # a 3D forward pass takes seconds on a CPU
        m.update(profile_inference(model, device, shape=(1, 1) + tuple(vol_ds.images.shape[1:]),
                                   warmup=min(5, reps), reps=reps))
        m["checkpoint"] = path
        m["prostate_errors"] = error_breakdown(ds2d, res)
        m["gland_extent"] = extent_breakdown(ds2d, res)
        per_seed.append(m)
        if j == 0:
            results["Improved U-Net 3D"], metrics["Improved U-Net 3D"] = res, m
            curves = os.path.join(os.path.dirname(path), "curves.png")
            if os.path.exists(curves):
                shutil.copy(curves, os.path.join(args.out, "curves_improved3d.png"))
        print(f"3D {path}: prostate Dice (2D slices) {m['pooled_dice']['prostate']:.4f} | volumetric "
              f"{native['volumetric_dice_full_res']['prostate']:.4f} | base/apex {m['prostate_dice_boundary']:.4f}"
              f" | mid {m['prostate_dice_mid']:.4f}")

    metrics["Improved U-Net 3D + largest component"] = lcc_first
    for name, m in metrics.items():
        res = results[name]
        m["dice_by_prostate_area"] = size_breakdown(ds2d, res)
        m["prostate_errors"] = error_breakdown(ds2d, res)
        m["gland_extent"] = extent_breakdown(ds2d, res)

    # resolution ceiling: a perfect prediction at the downsampled resolution, scored the same way
    oracle_res, oracle_native = evaluate_3d(_Oracle(vol_ds), args.study_root, vol_ds, ds2d, torch.device("cpu"))
    ceiling = summarise(oracle_res, positions)
    ceiling_summary = {"pooled_dice": ceiling["pooled_dice"], "prostate_dice_boundary": ceiling["prostate_dice_boundary"],
                       "prostate_dice_mid": ceiling["prostate_dice_mid"], **oracle_native}
    print(f"\ndownsampling ceiling (perfect 3D prediction at 1/2 resolution): prostate Dice "
          f"{ceiling['pooled_dice']['prostate']:.4f} | base/apex {ceiling['prostate_dice_boundary']:.4f} | "
          f"mid {ceiling['prostate_dice_mid']:.4f}")

    m3 = metrics["Improved U-Net 3D"]
    m3["worst_slices"] = plot_failures(ds2d, results["Improved U-Net 3D"], positions, "Improved U-Net 3D",
                                       os.path.join(args.out, "failures_improved3d.png"))
    plotted = [n for n in metrics if "largest component" not in n]  # figures show the trained models
    plot_autopsy_comparison(ds2d, {n: results[n] for n in plotted}, AUTOPSY_CASES,
                            os.path.join(args.out, "autopsy_comparison.png"))
    plot_examples(ds2d, {n: results[n]["preds"] for n in plotted}, os.path.join(args.out, "examples3d.png"))
    plot_position({n: metrics[n] for n in plotted}, os.path.join(args.out, "position_dice3d.png"))
    plot_size({n: metrics[n]["dice_by_prostate_area"] for n in plotted}, os.path.join(args.out, "size_dice3d.png"))

    def over_seeds(ms):
        rows = [headline(m) for m in ms]
        return {k: {"mean": st.mean(r[k] for r in rows), "std": st.stdev(r[k] for r in rows) if len(rows) > 1 else 0.0,
                    "runs": [r[k] for r in rows]} for k in rows[0]}

    seed_summary, seed_summary_lcc = over_seeds(per_seed), over_seeds(per_seed_lcc)
    save_json({"models": metrics, "improved3d_seeds": per_seed, "improved3d_seed_summary": seed_summary,
               "improved3d_lcc_seeds": per_seed_lcc, "improved3d_lcc_seed_summary": seed_summary_lcc,
               "downsampling_ceiling": ceiling_summary},
              os.path.join(args.out, "comparison3d.json"))

    print("\n| Test metric (same 540 slices) | " + " | ".join(metrics) + " |")
    print("| --- |" + " --- |" * len(metrics))
    for k in headline(m3):
        print(f"| {k} | " + " | ".join(f"{headline(m)[k]:.4g}" if isinstance(headline(m)[k], float)
                                       else str(headline(m)[k]) for m in metrics.values()) + " |")
    for c in CLASS_NAMES[1:]:
        print(f"| Dice {c} | " + " | ".join(f"{m['pooled_dice'][c]:.3f}" for m in metrics.values()) + " |")
    for label, summ in (("3D", seed_summary), ("3D + largest component", seed_summary_lcc)):
        print(f"\n{label} over {len(per_seed)} seed(s): " + "; ".join(
            f"{k} {v['mean']:.4g} ± {v['std']:.2g}" for k, v in summ.items()))
    print(f"\nfigures written to {args.out}/")


if __name__ == "__main__":
    main()
