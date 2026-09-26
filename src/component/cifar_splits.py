"""Build the CIFAR-100 split manifest.

The manifest holds a validation set taken from the official training set and
nested training budgets drawn from the remaining images. It is written once
and then read by every run, so all conditions use the same images.

Run from the repository root:

    python -m src.component.cifar_splits --data-root data \
        --output src/component/configs/splits/cifar100.json
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

SPLIT_SEED = 42
VALIDATION_PER_CLASS = 50
SHOTS_PER_CLASS = [5, 10, 20, 50]


def labels_sha256(labels: Sequence[int]) -> str:
    """Return a SHA-256 fingerprint of the label sequence.

    The manifest stores it so a run can check it reads the same dataset.
    """
    return hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()


def build_split_manifest(
    labels: Sequence[int],
    num_classes: int,
    validation_per_class: int,
    shots_per_class: Sequence[int],
    seed: int,
) -> dict:
    """Split image indices into validation and nested training budgets.

    Each class's indices are shuffled once with ``seed``. The first
    ``validation_per_class`` go to validation; the training budget of ``k``
    images per class takes the next ``k``, so every budget contains the
    smaller ones. The ``"full"`` budget holds every non-validation image.

    Args:
        labels: Class label of each image, indexed like the dataset.
        num_classes: Number of classes; every class must be present.
        validation_per_class: Validation images reserved per class.
        shots_per_class: Training budgets, in images per class.
        seed: Seed of the NumPy generator that shuffles the classes.

    Returns:
        Manifest with the settings, the label fingerprint, sorted
        ``validation_indices`` and ``train_indices`` keyed by budget.

    Raises:
        ValueError: If a class has too few images for validation plus the
            largest budget.
    """
    labels_array = np.asarray(labels)
    rng = np.random.default_rng(seed)
    needed = validation_per_class + max(shots_per_class)
    validation: list[int] = []
    budgets: dict[str, list[int]] = {str(shots): [] for shots in shots_per_class}
    budgets["full"] = []

    for label in range(num_classes):
        shuffled = rng.permutation(np.flatnonzero(labels_array == label))
        if len(shuffled) < needed:
            raise ValueError(f"class {label} has {len(shuffled)} images, needs {needed}")
        validation.extend(shuffled[:validation_per_class].tolist())
        remaining = shuffled[validation_per_class:]
        for shots in shots_per_class:
            budgets[str(shots)].extend(remaining[:shots].tolist())
        budgets["full"].extend(remaining.tolist())

    return {
        "split_seed": seed,
        "random_generator": "numpy PCG64",
        "validation_per_class": validation_per_class,
        "shots_per_class": list(shots_per_class),
        "labels_sha256": labels_sha256(labels),
        "validation_indices": sorted(validation),
        "train_indices": {name: sorted(indices) for name, indices in budgets.items()},
    }


def main() -> None:
    """Build the manifest from the CIFAR-100 training set and save it."""
    from torchvision.datasets import CIFAR100

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = CIFAR100(args.data_root, train=True, download=False)
    manifest = build_split_manifest(
        dataset.targets,
        len(dataset.classes),
        VALIDATION_PER_CLASS,
        SHOTS_PER_CLASS,
        SPLIT_SEED,
    )
    manifest = {
        "dataset": "CIFAR-100",
        "index_source": "torchvision CIFAR100(train=True)",
        "numpy_version": np.__version__,
        **manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sizes = {name: len(indices) for name, indices in manifest["train_indices"].items()}
    print(f"validation: {len(manifest['validation_indices'])}, training: {sizes}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
