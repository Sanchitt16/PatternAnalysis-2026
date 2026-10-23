"""
dataset.py - HipMRI 2D prostate slice loading, preprocessing and augmentation.

Data layout (HipMRI_Study_open/keras_slices_data):
    keras_slices_{train,validate,test}/case_XXX_week_Y_slice_Z.nii.gz       (MR image)
    keras_slices_seg_{train,validate,test}/seg_XXX_week_Y_slice_Z.nii.gz    (label map)

Labels: 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate.

Leakage: the provided splits are patient-disjoint (31 / 4 / 3 patients, verified by
`assert_patient_disjoint`), so every slice of a patient lives in exactly one split.

Preprocessing
    * Every slice is centre-cropped / zero-padded to 256 x 128 (2 of the training slices
      are 256 x 144).
    * Per-slice z-score intensity normalisation (as in Appendix B of the spec), because
      raw MR intensities are uncalibrated and vary between scans.
    * Masks are kept as integer label maps; one-hot encoding happens in the loss.

Loading ~12.6k Nifti files is slow, so the first run caches each split as a compressed
.npz next to the data. Later runs load the cache in a few seconds.

Augmentation (training only) runs on the GPU on whole batches (`augment_batch`):
small random rotation / scale / translation plus random intensity scale and shift.
Images are interpolated bilinearly and masks with nearest neighbour so labels stay valid.
No left-right flip: the rectum is posterior and the pelvis is roughly symmetric, but we
keep orientation fixed so boundary analysis stays anatomically consistent.
"""
import math
import os
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    raise ImportError("nibabel is required: pip install nibabel") from e

NUM_CLASSES = 6
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
PROSTATE = 5
TARGET_SHAPE = (256, 128)
SPLITS = ("train", "validate", "test")

_NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- file discovery
def parse_name(fname):
    """'case_004_week_0_slice_12.nii.gz' -> (patient=4, week=0, slice=12)."""
    m = _NAME_RE.search(os.path.basename(fname))
    if m is None:
        raise ValueError(f"Unexpected HipMRI filename: {fname}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def list_split(root, split):
    """Return a list of (image_path, mask_path, (patient, week, slice)) sorted by key."""
    img_dir = os.path.join(root, f"keras_slices_{split}")
    seg_dir = os.path.join(root, f"keras_slices_seg_{split}")
    if not os.path.isdir(img_dir) or not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"Missing {img_dir} or {seg_dir}")
    imgs = {parse_name(f): os.path.join(img_dir, f) for f in os.listdir(img_dir) if ".nii" in f}
    segs = {parse_name(f): os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if ".nii" in f}
    missing = set(imgs) ^ set(segs)
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} images without a matching mask (or vice versa)")
    return [(imgs[k], segs[k], k) for k in sorted(imgs)]


def assert_patient_disjoint(root):
    """Raise if any patient appears in more than one split (cross-patient leakage)."""
    pats = {s: {k[0] for _, _, k in list_split(root, s)} for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            shared = pats[a] & pats[b]
            if shared:
                raise RuntimeError(f"Patient leakage between {a} and {b}: {sorted(shared)}")
    return {s: sorted(p) for s, p in pats.items()}


# --------------------------------------------------------------------------- preprocessing
def _load_2d(path):
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 3:  # some HipMRI slices carry a trailing singleton dim
        arr = arr[:, :, 0]
    return arr


def crop_or_pad(arr, shape=TARGET_SHAPE):
    """Centre-crop or zero-pad a 2D array to `shape` (keeps image/mask alignment)."""
    out = np.zeros(shape, dtype=arr.dtype)
    h, w = arr.shape
    H, W = shape
    sh, sw = max((h - H) // 2, 0), max((w - W) // 2, 0)  # source offsets (crop)
    dh, dw = max((H - h) // 2, 0), max((W - w) // 2, 0)  # destination offsets (pad)
    ch, cw = min(h, H), min(w, W)
    out[dh:dh + ch, dw:dw + cw] = arr[sh:sh + ch, sw:sw + cw]
    return out


def zscore(img):
    return (img - img.mean()) / (img.std() + 1e-8)


def load_split_arrays(root, split, cache_dir=None, verbose=True):
    """Load one split as (images float16 [N,H,W], masks uint8 [N,H,W], keys int32 [N,3]).

    Results are cached to `<cache_dir>/hipmri2d_<split>.npz`.
    """
    cache_dir = cache_dir or root
    cache = os.path.join(cache_dir, f"hipmri2d_{split}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["images"], d["masks"], d["keys"]

    items = list_split(root, split)
    n = len(items)
    images = np.zeros((n,) + TARGET_SHAPE, dtype=np.float16)
    masks = np.zeros((n,) + TARGET_SHAPE, dtype=np.uint8)
    keys = np.zeros((n, 3), dtype=np.int32)
    for i, (ip, mp, k) in enumerate(items):
        img = crop_or_pad(_load_2d(ip).astype(np.float32))
        images[i] = zscore(img)
        msk = np.rint(_load_2d(mp)).astype(np.uint8)
        if msk.max() >= NUM_CLASSES:
            raise ValueError(f"{mp}: unexpected label {msk.max()}")
        masks[i] = crop_or_pad(msk)
        keys[i] = k
        if verbose and (i % 2000 == 0 or i == n - 1):
            print(f"  [{split}] loaded {i + 1}/{n}")
    try:
        np.savez_compressed(cache, images=images, masks=masks, keys=keys)
    except OSError as e:  # read-only data dir: fall back to no cache
        print(f"  (could not write cache {cache}: {e})")
    return images, masks, keys


# --------------------------------------------------------------------------- dataset
class HipMRISlices(Dataset):
    """In-memory HipMRI 2D slice dataset.

    __getitem__ returns (image [1,H,W] float32, mask [H,W] int64, key [3] int64).
    The key (patient, week, slice) is used for per-patient and base/apex analysis.
    """

    def __init__(self, root, split, cache_dir=None, limit=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.images, self.masks, self.keys = load_split_arrays(root, split, cache_dir)
        if limit:  # quick smoke tests
            self.images, self.masks, self.keys = self.images[:limit], self.masks[:limit], self.keys[:limit]
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i].astype(np.float32)).unsqueeze(0)
        msk = torch.from_numpy(self.masks[i].astype(np.int64))
        key = torch.from_numpy(self.keys[i].astype(np.int64))
        return img, msk, key


# --------------------------------------------------------------------------- slice position
def gland_position(keys, masks):
    """Label every slice by its position along the prostate.

    For each scan (patient, week) we take the slices that contain prostate, ordered by
    slice index, and split them into thirds: the two outer thirds are the gland
    boundary (base / apex, where contours are thin and low contrast) and the middle
    third is the mid-gland. Slices with no prostate are 'none'.

    Returns a numpy array of strings: 'boundary' | 'mid' | 'none'.
    """
    pos = np.array(["none"] * len(keys), dtype=object)
    has_p = (masks == PROSTATE).reshape(len(masks), -1).any(1)
    by_scan = defaultdict(list)
    for i, (p, w, s) in enumerate(keys):
        if has_p[i]:
            by_scan[(int(p), int(w))].append((int(s), i))
    for lst in by_scan.values():
        lst.sort()
        n = len(lst)
        lo, hi = math.floor(n / 3), math.ceil(2 * n / 3)
        for rank, (_, i) in enumerate(lst):
            pos[i] = "mid" if lo <= rank < hi else "boundary"
    return pos


# --------------------------------------------------------------------------- augmentation
def augment_batch(x, y, max_rot=10.0, scale=(0.9, 1.1), max_shift=0.05, intensity=0.1):
    """Random affine + intensity augmentation on a GPU batch.

    x: [B,1,H,W] float, y: [B,H,W] long. Returns augmented (x, y) on the same device.
    """
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot)
    s = torch.empty(b, device=dev).uniform_(*scale)
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    y = torch.nn.functional.grid_sample(y.unsqueeze(1).float(), grid, mode="nearest",
                                        padding_mode="zeros", align_corners=False).squeeze(1).long()
    gain = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * intensity
    bias = (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * intensity
    return x * gain + bias, y
