import hashlib
import json
from pathlib import Path

import numpy as np


def make_splits(targets, config, include_full=False):
    targets = np.asarray(targets)
    if targets.shape != (50000,) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Expected 50,000 integer training labels.")
    classes, counts = np.unique(targets, return_counts=True)
    if not np.array_equal(classes, np.arange(100)) or not np.all(counts == 500):
        raise ValueError("Expected classes 0-99 with 500 training images each.")

    validation_count = config["validation_per_class"]
    shots = config["shots_per_class"]
    if type(validation_count) is not int or not 1 <= validation_count < 500:
        raise ValueError("validation_per_class must be an integer from 1 to 499.")
    if not shots or any(type(k) is not int or k < 1 for k in shots):
        raise ValueError("shots_per_class must contain positive integers.")
    if len(set(shots)) != len(shots) or max(shots) + validation_count > 500:
        raise ValueError("Shot counts must be unique and fit beside validation data.")

    rng = np.random.Generator(np.random.PCG64(config["split_seed"]))
    validation = []
    training = {str(k): [] for k in sorted(shots)}
    full_training = []
    for class_id in range(100):
        indices = rng.permutation(np.flatnonzero(targets == class_id))
        validation.extend(indices[:validation_count].tolist())
        available = indices[validation_count:]
        if include_full:
            full_training.extend(available.tolist())
        for k in sorted(shots):
            training[str(k)].extend(available[:k].tolist())

    result = {
        "dataset": "CIFAR-100",
        "index_source": "torchvision CIFAR100(train=True)",
        "targets_sha256": hashlib.sha256(targets.astype("<i8").tobytes()).hexdigest(),
        "numpy_version": np.__version__,
        "random_generator": "PCG64",
        "split_seed": config["split_seed"],
        "validation_per_class": validation_count,
        "validation_indices": sorted(validation),
        "train_indices": {k: sorted(indices) for k, indices in training.items()},
        "test": {"source": "official test split", "size": 10000},
    }
    if include_full:
        result["full_train_indices"] = sorted(full_training)
    return result


if __name__ == "__main__":
    from torchvision.datasets import CIFAR100

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/cifar100_fewshot.json").read_text())
    dataset = CIFAR100(root=root / "data/cifar100", train=True, download=False)
    manifest = make_splits(dataset.targets, config)
    manifest["class_names"] = dataset.classes
    output = root / "configs/splits/cifar100_fewshot.json"
    content = json.dumps(manifest, indent=2) + "\n"
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise FileExistsError(f"{output} differs. Keep the existing split and review the changes.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Validation: {len(manifest['validation_indices'])} images")
    for shots, indices in manifest["train_indices"].items():
        print(f"{shots} shots: {len(indices)} training images")
    print(f"Manifest: {output}")
