"""
modules.py - network components and loss for HipMRI prostate segmentation (2D slices and 3D volumes).

Two models are compared on identical splits:

* UNet2D (baseline) - Ronneberger et al., "U-Net: Convolutional networks for biomedical
  image segmentation", MICCAI 2015. Double 3x3 conv blocks with BatchNorm + ReLU,
  max-pool downsampling, transposed-conv upsampling, concatenated skip connections.

* ImprovedUNet2D / ImprovedUNet3D - 2D and 3D versions of Isensee et al., "Brain tumor segmentation
  and radiomics survival prediction: Contribution to the BRATS 2017 challenge", 2018.
  Differences from the baseline, following Fig. 1 of that paper:
    - context modules: pre-activation residual blocks (norm -> lrelu -> conv x2) with
      spatial dropout (p=0.3) in between, summed with their input;
    - downsampling by 3x3 stride-2 convolutions instead of max pooling;
    - upsampling modules: nearest upsample followed by a 3x3 conv;
    - localisation modules: 3x3 conv then 1x1 conv that halves the channels;
    - deep supervision: 1x1 segmentation heads at three decoder scales, upsampled and
      summed element-wise into the final prediction;
    - instance normalisation and leaky ReLU (slope 0.01) throughout.

Everything here is pure PyTorch (no NumPy), as required by the spec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== baseline
class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x 2, the basic U-Net block."""

    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """Standard 2D U-Net baseline (4 down / 4 up levels)."""

    def __init__(self, in_channels=1, num_classes=6, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.decoders.append(DoubleConv(2 * f, f))
            ch = f
        self.head = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


# =========================================================================== improved U-Net
# The Improved U-Net blocks are written once and built in 2D or 3D through `dim`, so the 2D
# model (Normal tier) and the 3D model (Hard tier) share exactly the same architecture.
_CONV = {2: nn.Conv2d, 3: nn.Conv3d}
_NORM = {("instance", 2): nn.InstanceNorm2d, ("instance", 3): nn.InstanceNorm3d,
         ("batch", 2): nn.BatchNorm2d, ("batch", 3): nn.BatchNorm3d}
_DROP = {2: nn.Dropout2d, 3: nn.Dropout3d}


def _norm_act(ch, dim=2, norm="instance"):
    layer = _NORM[(norm, dim)](ch, affine=True) if norm == "instance" else _NORM[(norm, dim)](ch)
    return [layer, nn.LeakyReLU(0.01, inplace=True)]


class ContextModule(nn.Module):
    """Pre-activation residual block: x + conv(lrelu(norm(dropout(conv(lrelu(norm(x))))))."""

    def __init__(self, ch, p_drop=0.3, dim=2, norm="instance"):
        super().__init__()
        conv = _CONV[dim]
        self.body = nn.Sequential(
            *_norm_act(ch, dim, norm), conv(ch, ch, 3, padding=1),
            _DROP[dim](p_drop),
            *_norm_act(ch, dim, norm), conv(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class EncoderStage(nn.Module):
    """Conv (stride 1 for the first stage, stride 2 otherwise) followed by a context module.

    The conv output is the residual shortcut, as in Isensee et al. Fig. 1.
    """

    def __init__(self, cin, cout, stride, p_drop=0.3, dim=2, norm="instance"):
        super().__init__()
        self.conv = _CONV[dim](cin, cout, 3, stride=stride, padding=1)
        self.context = ContextModule(cout, p_drop, dim, norm)

    def forward(self, x):
        return self.context(self.conv(x))


class UpsampleModule(nn.Module):
    """Nearest-neighbour x2 upsample then 3x3(x3) conv that halves the channels."""

    def __init__(self, cin, cout, dim=2, norm="instance"):
        super().__init__()
        self.conv = nn.Sequential(*_norm_act(cin, dim, norm), _CONV[dim](cin, cout, 3, padding=1))

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class LocalisationModule(nn.Module):
    """3x3(x3) conv on the concatenated skip + upsampled features, then 1x1 conv halving channels."""

    def __init__(self, cin, cout, dim=2, norm="instance"):
        super().__init__()
        conv = _CONV[dim]
        self.body = nn.Sequential(
            *_norm_act(cin, dim, norm), conv(cin, cin // 2, 3, padding=1),
            *_norm_act(cin // 2, dim, norm), conv(cin // 2, cout, 1),
        )

    def forward(self, x):
        return self.body(x)


class ImprovedUNet(nn.Module):
    """Improved U-Net (Isensee et al. 2018) with residual context modules and deep supervision.

    dim=2 for slices [B,1,H,W], dim=3 for volumes [B,1,D,H,W]. With base=16 the encoder widths
    are 16, 32, 64, 128, 256 (5 levels), as in the paper. Every spatial size must be divisible
    by 16 (256 x 128 slices and 128 x 128 x 64 volumes are).
    """

    def __init__(self, in_channels=1, num_classes=6, base=16, p_drop=0.3, dim=2, norm="instance",
                 deep_supervision=True):
        super().__init__()
        self.dim = dim
        self.deep_supervision = deep_supervision
        conv = _CONV[dim]
        w = [base * 2 ** i for i in range(5)]  # 16, 32, 64, 128, 256
        self.enc = nn.ModuleList([EncoderStage(in_channels, w[0], 1, p_drop, dim, norm)] +
                                 [EncoderStage(w[i - 1], w[i], 2, p_drop, dim, norm) for i in range(1, 5)])

        # decoder, from the bottleneck (w[4]) upwards
        self.up = nn.ModuleList([UpsampleModule(w[i], w[i - 1], dim, norm) for i in range(4, 0, -1)])
        # localisation modules for levels 3, 2, 1 (the top level uses a plain conv)
        self.loc = nn.ModuleList([LocalisationModule(2 * w[i], w[i], dim, norm) for i in range(3, 0, -1)])
        self.top = nn.Sequential(*_norm_act(2 * w[0], dim, norm), conv(2 * w[0], 2 * w[0], 3, padding=1),
                                 *_norm_act(2 * w[0], dim, norm))
        # segmentation heads: levels 2 and 1 only exist with deep supervision
        if deep_supervision:
            self.seg2 = conv(w[2], num_classes, 1)
            self.seg1 = conv(w[1], num_classes, 1)
        self.seg0 = conv(2 * w[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
        x = skips.pop()  # bottleneck output (level 4)

        seg_maps = []
        for i, up in enumerate(self.up):
            x = torch.cat([up(x), skips.pop()], dim=1)  # concat with encoder level 3-i
            if i < 3:
                x = self.loc[i](x)
                if self.deep_supervision and i == 1:
                    seg_maps.append(self.seg2(x))  # level 2
                elif self.deep_supervision and i == 2:
                    seg_maps.append(self.seg1(x))  # level 1
            else:
                x = self.top(x)
                seg_maps.append(self.seg0(x))  # level 0 (full resolution)

        # deep supervision: upsample coarse maps and sum element-wise
        mode = "bilinear" if self.dim == 2 else "trilinear"
        out = seg_maps[0]
        for s in seg_maps[1:]:
            out = F.interpolate(out, size=s.shape[2:], mode=mode, align_corners=False) + s
        return out


class ImprovedUNet2D(ImprovedUNet):
    """2D Improved U-Net (Normal tier). The keyword options exist for the ablation study."""

    def __init__(self, in_channels=1, num_classes=6, base=16, p_drop=0.3, norm="instance",
                 deep_supervision=True):
        super().__init__(in_channels, num_classes, base, p_drop, dim=2, norm=norm,
                         deep_supervision=deep_supervision)


class ImprovedUNet3D(ImprovedUNet):
    """3D Improved U-Net (Hard tier) - same blocks with Conv3d / InstanceNorm3d / Dropout3d."""

    def __init__(self, in_channels=1, num_classes=6, base=16, p_drop=0.3):
        super().__init__(in_channels, num_classes, base, p_drop, dim=3)


# =========================================================================== loss
class DiceCELoss(nn.Module):
    """Soft multi-class Dice loss + cross-entropy.

    Dice is computed per class over the whole batch (so classes absent from one slice
    still contribute through the others), then averaged over the foreground classes.
    Cross-entropy keeps gradients well behaved early in training; Dice counteracts the
    extreme imbalance (prostate ~0.4% of pixels).
    """

    def __init__(self, num_classes=6, dice_weight=1.0, ce_weight=1.0, include_background=False,
                 class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.first = 0 if include_background else 1
        self.smooth = smooth
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target)
        probs = logits.float().softmax(1)
        onehot = F.one_hot(target, self.num_classes).movedim(-1, 1).float()  # [B,C,...] (2D or 3D)
        dims = (0,) + tuple(range(2, logits.dim()))
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        dice = dice[self.first:]
        if self.class_weights is not None:
            w = self.class_weights[self.first:].to(dice.device)
            dice_loss = 1 - (dice * w).sum() / w.sum()
        else:
            dice_loss = 1 - dice.mean()
        return self.dice_weight * dice_loss + self.ce_weight * ce


# =========================================================================== factory
MODELS = {"unet": UNet2D, "improved": ImprovedUNet2D, "improved3d": ImprovedUNet3D}

# Ablation of the 2D Improved U-Net: each variant switches off one suspected cause of its
# lower accuracy versus the plain U-Net ("noall" switches off all three at once).
ABLATIONS = {
    "improved_nodrop": dict(p_drop=0.0),                 # A: no spatial dropout
    "improved_bn": dict(norm="batch"),                   # B: BatchNorm instead of InstanceNorm
    "improved_nods": dict(deep_supervision=False),       # C: no deep supervision
    "improved_noall": dict(p_drop=0.0, norm="batch", deep_supervision=False),  # D: A + B + C
}
MODEL_CHOICES = list(MODELS) + list(ABLATIONS)


def build_model(name, num_classes=6):
    if name in ABLATIONS:
        return ImprovedUNet2D(in_channels=1, num_classes=num_classes, **ABLATIONS[name])
    if name not in MODELS:
        raise ValueError(f"model must be one of {MODEL_CHOICES}")
    return MODELS[name](in_channels=1, num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
