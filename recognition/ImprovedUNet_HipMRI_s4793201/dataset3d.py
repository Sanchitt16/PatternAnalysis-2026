"""
dataset3d.py - HipMRI 3D volume loading, downsampling, splitting and augmentation (Hard tier).

Data layout (HipMRI_Study_open):
    semantic_MRs/<ID>_Week<w>_LFOV.nii.gz            (MR volume, 256 x 256 x 128, ~1.56 mm)
    semantic_labels_only/<ID>_Week<w>_SEMANTIC.nii.gz (labels 0-5, same grid)
    keras_slices_data/...                             (2D slices; used only for the split map)

Splits. Every 3D scan inherits the split of its patient in the provided 2D split
(31 / 4 / 3 patients, patient-disjoint). This keeps the 3D test set to the same patients
(40-42) as the 2D experiments, so the 2D and 3D models are compared on the same people and
no patient leaks between experiments. Patients that have no 2D slices go to training.

Downsampling (x2 per axis -> 128 x 128 x 64, ~3.1 mm voxels; the spec's "downsampled HipMRI
3D"). Images: 2x2x2 average pooling. Masks: majority vote over each 2x2x2 block (average-pool
the one-hot mask, then argmax). Plain stride-2 subsampling would delete thin structures
such as the prostate apex, so majority voting matters here.

Shape: every volume is cropped/padded to 256 x 256 x 128 first (one volume has 144 slices;
its crop is centred on the prostate). In-plane voxel size varies between scans (1.41-1.88 mm,
mostly 1.68 mm); volumes are not resampled, matching how the provided 2D slices were made.

Link to the 2D data: 2D slice k of a scan is exactly volume[:, 98 + k, :] flipped along the
first axis (verified pixel-for-pixel), i.e. the 2D set is the central 60 slices along axis 1.
predict3d.py uses this to score the 3D model on exactly the same pixels as the 2D models.

Preprocessing: per-volume z-score on the downsampled image. Results are cached as
`hipmri3d_<split>.npz` (float16 images, uint8 masks) so later runs start in seconds.

Augmentation (training, on the GPU): small random 3D rotation about the slice axis, scale,
translation, and intensity gain/shift; nearest-neighbour for masks.
"""
import math
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    raise ImportError("nibabel is required: pip install nibabel") from e

NUM_CLASSES = 6
PROSTATE = 5
SPLITS = ("train", "validate", "test")
FACTOR = 2
_VOL = re.compile(r"([A-Za-z]*)(\d+)_Week(\d+)", re.IGNORECASE)
_SLICE = re.compile(r"case_(\d+)_week_(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- discovery and splits
def _index(folder):
    """{(patient:int, week:int): path} for every Nifti in folder."""
    out = {}
    for f in sorted(os.listdir(folder)):
        m = _VOL.match(f)
        if m and ".nii" in f:
            out[(int(m.group(2)), int(m.group(3)))] = os.path.join(folder, f)
    return out


def patient_split_map(root):
    """patient -> split, read from the 2D keras_slices folder names."""
    k2 = os.path.join(root, "keras_slices_data")
    out = {}
    for s in SPLITS:
        for f in os.listdir(os.path.join(k2, f"keras_slices_{s}")):
            m = _SLICE.match(f)
            if m:
                p = int(m.group(1))
                if out.get(p, s) != s:
                    raise RuntimeError(f"patient {p} appears in two 2D splits")
                out[p] = s
    return out


def list_volumes(root):
    """Return {split: [(image_path, label_path, (patient, week)), ...]} with patient-level splits."""
    mrs = _index(os.path.join(root, "semantic_MRs"))
    lbs = _index(os.path.join(root, "semantic_labels_only"))
    if set(mrs) != set(lbs):
        raise RuntimeError(f"{len(set(mrs) ^ set(lbs))} volumes without a matching label (or vice versa)")
    pmap = patient_split_map(root)
    out = defaultdict(list)
    for k in sorted(mrs):
        out[pmap.get(k[0], "train")].append((mrs[k], lbs[k], k))
    pats = {s: {k[0] for _, _, k in out[s]} for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if pats[a] & pats[b]:
                raise RuntimeError(f"patient leakage between {a} and {b}: {sorted(pats[a] & pats[b])}")
    return out


# --------------------------------------------------------------------------- preprocessing
def _load(path):
    a = np.asanyarray(nib.load(path).dataobj)
    if a.ndim == 4:
        a = a[..., 0]
    return a


def downsample_image(img, factor=FACTOR):
    t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None]
    return F.avg_pool3d(t, factor)[0, 0].numpy()


def downsample_mask(msk, factor=FACTOR, num_classes=NUM_CLASSES):
    """Majority vote over factor^3 blocks (ties go to the higher label, i.e. towards organs)."""
    t = torch.from_numpy(np.rint(msk).astype(np.int64))
    oh = F.one_hot(t, num_classes).permute(3, 0, 1, 2).float()[None]
    votes = F.avg_pool3d(oh, factor)[0]
    votes += torch.arange(num_classes).view(-1, 1, 1, 1) * 1e-4  # deterministic tie-break
    return votes.argmax(0).to(torch.uint8).numpy()


FULL_SHAPE = (256, 256, 128)


def fit_volume(img, msk, shape=FULL_SHAPE):
    """Crop/zero-pad a full-resolution (image, mask) pair to `shape`.

    One HipMRI volume has 144 instead of 128 slices along the last axis; the crop window is
    centred on the prostate (label 5) so the gland is never cut off.
    """
    out_i, out_m = np.zeros(shape, np.float32), np.zeros(shape, np.uint8)
    src, dst = [], []
    for ax, (n, N) in enumerate(zip(img.shape, shape)):
        if n > N:
            p = np.where((msk == PROSTATE).any(axis=tuple(a for a in range(3) if a != ax)))[0]
            centre = int(p.mean()) if len(p) else n // 2
            start = int(np.clip(centre - N // 2, 0, n - N))
            src.append(slice(start, start + N)); dst.append(slice(0, N))
        else:
            off = (N - n) // 2
            src.append(slice(0, n)); dst.append(slice(off, off + n))
    out_i[tuple(dst)] = img[tuple(src)]
    out_m[tuple(dst)] = msk[tuple(src)]
    return out_i, out_m


def pad_to_multiple(a, m=16):
    """Zero-pad every axis to a multiple of m (the 4 down-samplings need /16)."""
    pad = [(0, (-s) % m) for s in a.shape]
    return np.pad(a, pad) if any(p for _, p in pad) else a


def load_split(root, split, cache_dir=None, verbose=True):
    """(images float16 [N,X,Y,Z], masks uint8 [N,X,Y,Z], keys int32 [N,2]) at 1/FACTOR resolution."""
    cache = os.path.join(cache_dir or root, f"hipmri3d_{split}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["images"], d["masks"], d["keys"]
    items = list_volumes(root)[split]
    imgs, msks, keys = [], [], []
    for i, (ip, lp, k) in enumerate(items):
        full_img, full_msk = fit_volume(_load(ip).astype(np.float32), np.rint(_load(lp)).astype(np.uint8))
        img = downsample_image(full_img)
        img = (img - img.mean()) / (img.std() + 1e-8)
        msk = downsample_mask(full_msk)
        if img.shape != msk.shape:
            raise ValueError(f"{k}: image {img.shape} vs mask {msk.shape}")
        imgs.append(pad_to_multiple(img).astype(np.float16))
        msks.append(pad_to_multiple(msk))
        keys.append(k)
        if verbose and (i % 20 == 0 or i == len(items) - 1):
            print(f"  [{split}] {i + 1}/{len(items)} volumes")
    shapes = {a.shape for a in imgs}
    if len(shapes) != 1:
        raise ValueError(f"{split}: volumes have different shapes after downsampling: {shapes}")
    out = np.stack(imgs), np.stack(msks), np.array(keys, dtype=np.int32)
    try:
        np.savez_compressed(cache, images=out[0], masks=out[1], keys=out[2])
    except OSError as e:
        print(f"  (could not write cache {cache}: {e})")
    return out


class HipMRIVolumes(Dataset):
    """In-memory downsampled HipMRI volumes. Returns (image [1,X,Y,Z], mask [X,Y,Z], key [2])."""

    def __init__(self, root, split, cache_dir=None, limit=None):
        self.images, self.masks, self.keys = load_split(root, split, cache_dir)
        if limit:
            self.images, self.masks, self.keys = self.images[:limit], self.masks[:limit], self.keys[:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return (torch.from_numpy(self.images[i].astype(np.float32))[None],
                torch.from_numpy(self.masks[i].astype(np.int64)),
                torch.from_numpy(self.keys[i].astype(np.int64)))


# --------------------------------------------------------------------------- augmentation
def augment_batch3d(x, y, max_rot=10.0, scale=(0.9, 1.1), max_shift=0.05, intensity=0.1):
    """Random in-plane rotation (about the last, slice axis), isotropic scale, shift and intensity.

    x: [B,1,X,Y,Z] float, y: [B,X,Y,Z] long.
    """
    b, dev = x.shape[0], x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot)
    s = torch.empty(b, device=dev).uniform_(*scale)
    t = (torch.rand(b, 3, device=dev) * 2 - 1) * max_shift * 2
    c, sn = torch.cos(ang) / s, torch.sin(ang) / s
    z, o = torch.zeros_like(c), torch.ones_like(c) / s
    # affine_grid uses (x, y, z) = (last, middle, first) tensor axes; rotate the two in-plane axes
    theta = torch.stack([torch.stack([o, z, z, t[:, 0]], 1),
                         torch.stack([z, c, -sn, t[:, 1]], 1),
                         torch.stack([z, sn, c, t[:, 2]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y = F.grid_sample(y.unsqueeze(1).float(), grid, mode="nearest", padding_mode="zeros",
                      align_corners=False).squeeze(1).long()
    gain = 1 + (torch.rand(b, 1, 1, 1, 1, device=dev) * 2 - 1) * intensity
    bias = (torch.rand(b, 1, 1, 1, 1, device=dev) * 2 - 1) * intensity
    return x * gain + bias, y
