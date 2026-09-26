"""CIFAR-100 data pipeline.

Images stay as 32x32 uint8 tensors on the CPU, where RandAugment is applied.
``to_model_input`` resizes and normalizes whole batches on the training
device. The training order is a fixed function of the seed and the step, so
an interrupted run resumes with exactly the same batches and augmentations.
"""

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RANDAUGMENT_OPS = 2
RANDAUGMENT_MAGNITUDE = 9


def load_manifest(path: Path) -> dict:
    """Read a split manifest written by ``cifar_splits``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def budget_indices(manifest: dict, budget: str) -> list[int]:
    """Return the training image indices of one budget, e.g. ``"5"`` or ``"full"``."""
    budgets = manifest["train_indices"]
    if budget not in budgets:
        raise ValueError(f"unknown budget {budget!r}; available: {sorted(budgets)}")
    return budgets[budget]


class CifarImages(Dataset):
    """A subset of CIFAR images, returned as uint8 tensors with label and image ID.

    Items are addressed by ``(position, visit_seed)``. With RandAugment on,
    the augmentation of that visit depends only on ``visit_seed``, not on the
    worker process or on earlier samples.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: Sequence[int],
        indices: Sequence[int],
        randaugment: bool,
    ) -> None:
        """
        Args:
            images: All images of the split, shape (N, 32, 32, 3), uint8.
            labels: Class label of every image in ``images``.
            indices: Image IDs (rows of ``images``) in this subset.
            randaugment: Whether to apply RandAugment to each visit.
        """
        self.images = images
        self.labels = labels
        self.indices = list(indices)
        self.augment = (
            v2.RandAugment(num_ops=RANDAUGMENT_OPS, magnitude=RANDAUGMENT_MAGNITUDE)
            if randaugment
            else None
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
        position, visit_seed = item
        image_id = self.indices[position]
        image = torch.from_numpy(self.images[image_id]).permute(2, 0, 1).contiguous()  # (3, 32, 32)
        if self.augment is not None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(visit_seed)
                image = self.augment(image)
        return image, int(self.labels[image_id]), image_id


class StepBatchSampler(Sampler[list[tuple[int, int]]]):
    """Yield the training batches for steps ``start_step`` to ``end_step - 1``.

    The sample stream is a sequence of epochs; epoch ``e`` is a permutation
    of all positions drawn from ``(seed, e)``. Step ``s`` takes stream items
    ``s * batch_size`` to ``(s + 1) * batch_size - 1``, so batches may cross
    epoch boundaries and a run started at any step sees the same stream.
    """

    def __init__(
        self, num_items: int, batch_size: int, seed: int, start_step: int, end_step: int
    ) -> None:
        self.num_items = num_items
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.end_step = end_step

    def __len__(self) -> int:
        return self.end_step - self.start_step

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        epoch, order = -1, np.empty(0, dtype=np.int64)
        for step in range(self.start_step, self.end_step):
            batch = []
            for k in range(step * self.batch_size, (step + 1) * self.batch_size):
                if k // self.num_items != epoch:
                    epoch = k // self.num_items
                    order = np.random.default_rng([self.seed, epoch]).permutation(self.num_items)
                batch.append((int(order[k % self.num_items]), self.seed * 1_000_003 + k))
            yield batch


def evaluation_batches(num_items: int, batch_size: int) -> list[list[tuple[int, int]]]:
    """Return all positions in order, in batches, without augmentation seeds."""
    return [
        [(position, 0) for position in range(start, min(start + batch_size, num_items))]
        for start in range(0, num_items, batch_size)
    ]


def to_model_input(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize uint8 images (B, 3, H, W) to (B, 3, size, size) and normalize.

    Bilinear upsampling, then ImageNet mean and standard deviation.
    """
    x = images.float() / 255
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def load_cifar100(root: Path, train: bool) -> tuple[np.ndarray, list[int]]:
    """Load CIFAR-100 images (N, 32, 32, 3) uint8 and labels from ``root``."""
    from torchvision.datasets import CIFAR100

    dataset = CIFAR100(root, train=train, download=False)
    return dataset.data, list(dataset.targets)
