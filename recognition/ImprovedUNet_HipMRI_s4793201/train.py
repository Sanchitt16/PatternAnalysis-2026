"""
train.py - train, validate, test and save a HipMRI 2D segmentation model.

Examples (run from this folder):
    # proposed model
    python train.py --model improved --data-root ../../../data/keras_slices_data
    # baseline on the identical splits
    python train.py --model unet --data-root ../../../data/keras_slices_data
    # 2-minute pipeline check
    python train.py --model improved --epochs 2 --limit 256 --data-root <path>
    # Hard tier: 3D Improved U-Net on downsampled volumes (data root = HipMRI_Study_open)
    python train.py --model improved3d --data-root ../../../data --epochs 150 --batch-size 2

Outputs (in --out, default runs/<model>/):
    best.pt            weights with the best validation prostate Dice
    history.csv/.json  per-epoch train loss, val loss, val Dice per class
    curves.png         loss and Dice curves
    test_metrics.json  pooled Dice/IoU per class, per-patient Dice, base/apex vs mid-gland,
                       false-alarm rate, parameter count, peak training VRAM, latency

Training recipe (same for both models so the comparison is fair):
    AdamW (lr 5e-4, weight decay 1e-5), cosine decay with 1-epoch linear warm-up,
    Dice + cross-entropy loss, batch 16, GPU augmentation, mixed precision on CUDA,
    model selection on the validation split only; the test split is touched once at the end.
"""
import argparse
import csv
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import (CLASS_NAMES, HipMRISlices, assert_patient_disjoint, augment_batch,
                     gland_position)
from dataset3d import HipMRIVolumes, augment_batch3d, list_volumes
from modules import DiceCELoss, build_model, count_parameters
from utils import (get_device, plot_history, predict_split, profile_inference, save_json,
                   set_seed, summarise)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["unet", "improved", "improved3d"], default="improved")
    ap.add_argument("--data-root", required=True,
                    help="2D: path to keras_slices_data; 3D: path to the folder holding semantic_MRs, "
                         "semantic_labels_only and keras_slices_data")
    ap.add_argument("--cache-dir", default=None, help="where to cache .npz splits (default: data root)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="use only N slices/volumes per split (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def is_3d(args):
    return args.model.endswith("3d")


def make_loaders(args):
    if is_3d(args):
        list_volumes(args.data_root)  # raises on patient leakage or unmatched labels
        sets = {s: HipMRIVolumes(args.data_root, s, args.cache_dir, args.limit) for s in ("train", "validate", "test")}
        eval_bs = 1
    else:
        assert_patient_disjoint(args.data_root)
        sets = {s: HipMRISlices(args.data_root, s, args.cache_dir, args.limit) for s in ("train", "validate", "test")}
        eval_bs = 32
    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(sets["train"], args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.workers, pin_memory=pin),
        "validate": DataLoader(sets["validate"], eval_bs, shuffle=False, num_workers=args.workers, pin_memory=pin),
        "test": DataLoader(sets["test"], eval_bs, shuffle=False, num_workers=args.workers, pin_memory=pin),
    }
    return sets, loaders


@torch.no_grad()
def validation_loss(model, loader, loss_fn, device, amp):
    model.eval()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=amp):
            total += loss_fn(model(x), y).item() * len(x)
        n += len(x)
    return total / max(n, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    amp = device.type == "cuda"
    out = args.out or os.path.join("runs", args.model)
    os.makedirs(out, exist_ok=True)
    print(f"device={device}  model={args.model}  out={out}")

    sets, loaders = make_loaders(args)
    print({s: len(d) for s, d in sets.items()})

    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"parameters: {n_params / 1e6:.2f} M")

    loss_fn = DiceCELoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loaders["train"])
    total_steps = args.epochs * steps_per_epoch
    warmup = steps_per_epoch

    def lr_lambda(step):  # linear warm-up for 1 epoch, then cosine decay to 0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best, t_start = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y, _ in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if not args.no_augment:
                x, y = augment_batch3d(x, y) if is_3d(args) else augment_batch(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(x)
            seen += len(x)

        val = summarise(predict_split(model, loaders["validate"], device, amp))
        vd = val["pooled_dice"]
        row = {"epoch": epoch,
               "train_loss": round(run_loss / max(seen, 1), 5),
               "val_loss": round(validation_loss(model, loaders["validate"], loss_fn, device, amp), 5),
               "val_dice_prostate": vd["prostate"],
               "val_dice_mean_fg": round(sum(vd[c] for c in CLASS_NAMES[1:]) / 5, 4),
               **{f"val_dice_{c}": vd[c] for c in CLASS_NAMES[1:5]},
               "lr": opt.param_groups[0]["lr"],
               "epoch_s": round(time.time() - t0, 1)}
        history.append(row)
        flag = ""
        if row["val_dice_prostate"] > best:
            best = row["val_dice_prostate"]
            torch.save({"model": args.model, "state_dict": model.state_dict(), "epoch": epoch,
                        "val_dice_prostate": best}, os.path.join(out, "best.pt"))
            flag = "  * saved"
        print(f"ep {epoch:3d}/{args.epochs}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
              f"prostate {row['val_dice_prostate']:.4f}  mean-fg {row['val_dice_mean_fg']:.4f}  "
              f"{row['epoch_s']:.0f}s{flag}")

        with open(os.path.join(out, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)
        save_json(history, os.path.join(out, "history.json"))
        plot_history(history, os.path.join(out, "curves.png"), f"{args.model} on HipMRI 2D")

    train_time_min = (time.time() - t_start) / 60
    train_vram = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")

    # ---- final, single evaluation on the held-out test split with the best checkpoint
    ckpt = torch.load(os.path.join(out, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_ds = sets["test"]
    res = predict_split(model, loaders["test"], device, amp)
    # slice-position analysis needs per-slice counts; for 3D it is done in predict3d.py
    positions = None if is_3d(args) else gland_position(test_ds.keys, test_ds.masks)
    metrics = summarise(res, positions=positions)
    metrics.update({"model": args.model, "best_epoch": ckpt["epoch"],
                    "best_val_dice_prostate": ckpt["val_dice_prostate"],
                    "parameters": n_params, "train_time_min": round(train_time_min, 1),
                    "train_peak_vram_mb": round(train_vram, 1), "epochs": args.epochs,
                    "batch_size": args.batch_size, "device": torch.cuda.get_device_name(0) if amp else "cpu"})
    metrics.update(profile_inference(model, device, shape=(1, 1) + tuple(test_ds.images.shape[1:])))
    save_json(metrics, os.path.join(out, "test_metrics.json"))

    print("\n=== TEST (best checkpoint, epoch", ckpt["epoch"], ")")
    for c in CLASS_NAMES:
        print(f"  {c:<10} Dice {metrics['pooled_dice'][c]:.4f}  IoU {metrics['pooled_iou'][c]:.4f}")
    if positions is not None:
        print(f"  prostate Dice boundary {metrics['prostate_dice_boundary']:.4f} | mid {metrics['prostate_dice_mid']:.4f}")
    print(f"  false-alarm rate on prostate-free slices {metrics['false_alarm_rate']:.3f}")
    print(f"  per patient {metrics['per_patient_prostate_dice']}")
    print(f"  params {n_params / 1e6:.2f} M | train VRAM {train_vram:.0f} MB | "
          f"latency {metrics['latency_ms_per_slice']} ms/{'volume' if is_3d(args) else 'slice'} | {train_time_min:.1f} min")


if __name__ == "__main__":
    main()
