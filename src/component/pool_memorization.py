"""Memorization check for a generated pool: are its images near copies of the training images?

A LoRA adapter fitted on a few images per class can learn to reproduce them.
Its pool would then add duplicates of the real training images, not new
images, and a gain from it would not be a gain from generation. This check
compares every generated image with the training images of its own class
(the images its adapter was fitted on) in the feature space of an
ImageNet-pretrained ResNet-50 (cosine similarity of the pooled features):

- generated: for each generated image, the similarity to its nearest training
  image of the same class;
- held out: the same for each validation image, which the adapter never saw.
  This is how close a new real image of the class comes to the training set.

A pool that copies its training images has many generated images closer to
a training image than almost any held-out real image is. The summary reports
the medians of both, the share of generated images above the 99th percentile
of the held-out similarities, and the share above a cosine of 0.95, and the
figure shows the most similar generated/training pairs for a visual check.

    python -m src.component.pool_memorization --dataset ham10000 --budget 5 \\
        --pool data/synthetic/sd15_lora_ham10000_b5/pool.npz \\
        --figure reports/Latex_report/fig/sd15_lora_ham10000_b5_nearest

The summary is saved as ``memorization.json`` next to the pool.
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

NEAR_COPY = 0.95  # cosine similarity treated as a near copy
QUANTILE = 0.99


def nearest_similarity(query: np.ndarray, query_labels: Sequence[int], reference: np.ndarray,
                       reference_labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """For each query, the cosine similarity to and index of its nearest reference of the same class.

    Features are L2-normalized here, so any scale is accepted.

    Raises:
        ValueError: If a query class has no reference image.
    """
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    query_labels, reference_labels = np.asarray(query_labels), np.asarray(reference_labels)
    similarity = np.empty(len(query))
    nearest = np.empty(len(query), dtype=np.int64)
    for label in np.unique(query_labels):
        q = np.flatnonzero(query_labels == label)
        r = np.flatnonzero(reference_labels == label)
        if len(r) == 0:
            raise ValueError(f"class {label} has no reference image")
        sims = query[q] @ reference[r].T
        similarity[q] = sims.max(axis=1)
        nearest[q] = r[sims.argmax(axis=1)]
    return similarity, nearest


def summarize(generated: np.ndarray, held_out: np.ndarray, near_copy: float = NEAR_COPY) -> dict:
    """Medians, the held-out 99th percentile, and the shares of generated images above it and above ``near_copy``."""
    threshold = float(np.quantile(held_out, QUANTILE))
    return {
        "generated_images": len(generated),
        "held_out_images": len(held_out),
        "generated_median": float(np.median(generated)),
        "held_out_median": float(np.median(held_out)),
        "held_out_p99": threshold,
        "share_above_held_out_p99": float(np.mean(generated > threshold)),
        "near_copy_threshold": near_copy,
        "share_near_copy": float(np.mean(generated > near_copy)),
    }


def verdict(summary: dict) -> str:
    """One line for the log: no sign of copying, or how many images look copied."""
    share = summary["share_above_held_out_p99"]
    if share <= 0.05 and summary["share_near_copy"] <= 0.01:
        return "no sign of copying"
    return (f"WARNING: {share:.0%} of generated images are closer to a training image than 99% of held-out images; "
            "look at the nearest-pair figure")


def features(images: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    """Pooled ImageNet ResNet-50 features of uint8 RGB images (any size; resized to 224)."""
    import torch
    import torch.nn.functional as F
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(np.array(images[start:start + batch_size])).to(device).permute(0, 3, 1, 2).float() / 255
            if x.shape[-1] != 224:
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
            out.append(model((x - mean) / std).float().cpu().numpy())
    return np.concatenate(out)


def plot_pairs(generated: np.ndarray, training: np.ndarray, similarity: np.ndarray, out_stem: Path,
               pairs: int = 8) -> None:
    """The ``pairs`` most similar generated/training pairs, generated above and training below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-similarity)[:pairs]
    fig, axes = plt.subplots(2, len(order), figsize=(1.6 * len(order), 3.6), squeeze=False)
    for col, i in enumerate(order):
        axes[0][col].imshow(generated[i])
        axes[0][col].set_title(f"{similarity[i]:.3f}", fontsize=8)
        axes[1][col].imshow(training[i])
    for ax in axes.ravel():
        ax.axis("off")
    axes[0][0].text(-0.1, 0.5, "generated", transform=axes[0][0].transAxes, rotation=90, ha="right", va="center")
    axes[1][0].text(-0.1, 0.5, "nearest\ntraining", transform=axes[1][0].transAxes, rotation=90, ha="right",
                    va="center")
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["flowers102", "ham10000"], required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--figure", type=Path, help="path without extension for the nearest-pair figure")
    parser.add_argument("--data-root", type=Path, help="default: data/<dataset>")
    parser.add_argument("--manifest", type=Path, help="default: src/component/configs/splits/<dataset>.json")
    args = parser.parse_args()

    import torch

    from src.component.data import budget_indices, load_manifest
    from src.component.stage3_data import load_images
    from src.component.synthetic_pool import load_pool

    images, labels = load_images(args.data_root or Path("data") / args.dataset)
    manifest = load_manifest(args.manifest or Path("src/component/configs/splits") / f"{args.dataset}.json")
    train = budget_indices(manifest, args.budget)
    validation = manifest["validation_indices"]
    pool_images, pool_labels, _ = load_pool(args.pool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_images = np.array(images[train])
    train_labels = [labels[i] for i in train]
    train_features = features(train_images, device)
    generated, nearest = nearest_similarity(features(pool_images, device), pool_labels, train_features, train_labels)
    held_out, _ = nearest_similarity(features(np.array(images[validation]), device), [labels[i] for i in validation],
                                     train_features, train_labels)
    summary = {"dataset": args.dataset, "budget": args.budget, "pool": str(args.pool),
               "feature_model": "torchvision ResNet-50, IMAGENET1K_V2, pooled features, cosine similarity",
               **summarize(generated, held_out)}
    summary["verdict"] = verdict(summary)
    (args.pool.parent / "memorization.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.figure:
        plot_pairs(pool_images, train_images[nearest], generated, args.figure)
    print(f"{args.pool.parent.name}: nearest-training similarity, generated median {summary['generated_median']:.3f} "
          f"vs held-out median {summary['held_out_median']:.3f}; "
          f"{summary['share_above_held_out_p99']:.1%} above the held-out 99th percentile "
          f"({summary['held_out_p99']:.3f}); {summary['share_near_copy']:.1%} above {NEAR_COPY}: {summary['verdict']}")


if __name__ == "__main__":
    main()
