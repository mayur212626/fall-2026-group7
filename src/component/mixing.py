"""Batch-level Mixup and CutMix for the conventional-augmentation conditions.

Each training batch is mixed with a shuffled copy of itself: Mixup blends
whole images, CutMix pastes a box from the partner image. The targets mix
the two one-hot labels with the same weight (for CutMix, the share of the
image area that stays the original), so the loss uses soft targets. Settings
follow the torchvision classification reference: Mixup alpha 0.2 and CutMix
alpha 1.0, applied to every batch. Mixing happens on the 32x32 images, where
RandAugment is also applied, before upsampling.

The random draws of a batch depend only on the run seed and the step, so a
resumed run mixes exactly as an uninterrupted one, and the global torch
random state is never touched.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

MIXING = {"mixup": {"alpha": 0.2}, "cutmix": {"alpha": 1.0}}
# Third seed word of the mixing stream. Fixed, so the draws never change when methods are added, and
# distinct from the two-word (seed, epoch) streams of the batch sampler.
MIXING_STREAM = 2


def mix_batch(images: torch.Tensor, labels: torch.Tensor, method: str, seed: int, step: int,
              num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix a batch with a shuffled copy of itself.

    Args:
        images: Batch (B, 3, H, W), uint8 or float on the 0-255 scale.
        labels: Class labels (B,).
        method: ``"mixup"`` or ``"cutmix"``.
        seed: Run seed.
        step: Optimizer step of this batch.
        num_classes: Number of classes for the soft targets.

    Returns:
        The mixed images as float on the 0-255 scale, and soft targets (B, num_classes).

    Raises:
        ValueError: If ``method`` is not a mixing condition.
    """
    if method not in MIXING:
        raise ValueError(f"unknown mixing method {method!r}; expected one of {sorted(MIXING)}")
    rng = np.random.default_rng([seed, step, MIXING_STREAM])
    lam = float(rng.beta(MIXING[method]["alpha"], MIXING[method]["alpha"]))
    partner = torch.from_numpy(rng.permutation(len(labels)))
    x = images.float()
    if method == "mixup":
        mixed = lam * x + (1 - lam) * x[partner]
    else:
        height, width = x.shape[-2:]
        cut = math.sqrt(1 - lam)
        cut_h, cut_w = int(height * cut), int(width * cut)
        cy, cx = int(rng.integers(height)), int(rng.integers(width))
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[partner][:, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (height * width)
    one_hot = F.one_hot(labels.long(), num_classes).float()
    return mixed, lam * one_hot + (1 - lam) * one_hot[partner]
