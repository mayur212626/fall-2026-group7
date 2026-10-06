"""Quality of a generated image set against a real reference set.

Both measures work on image features (one row per image), so every
generator is scored the same way:

- the Fréchet distance between Gaussians fitted to the two feature sets
  (FID when the features come from the standard Inception network), and
- Improved Precision & Recall (Kynkäänniemi et al., 2019): precision is the
  fraction of generated images that fall inside the real manifold, recall the
  fraction of real images inside the generated manifold. Each manifold is the
  union of balls around its points, with radius the distance to the k-th
  nearest neighbour in the same set.

Features are clean-fid's Inception features in its "clean" mode: every
32x32 image is resized to 299x299 with PIL bicubic before the network. The
command line scores a folder of generated 32x32 PNGs against the real
training images of one budget (by default 50 per class, 5,000 images), and
reports the same scores for a second, disjoint set of real training images
with the same classes, as the value a perfect generator would reach:

    python -m src.component.image_quality --generated data/synthetic/sd15_prompt \
        --output reports/Latex_report/tables/quality_sd15_prompt.json
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """Square root of a symmetric positive semi-definite matrix."""
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to feature sets ``a`` and ``b``.

    ``|mu_a - mu_b|^2 + Tr(S_a + S_b - 2 (S_a S_b)^(1/2))``, with the trace of
    the matrix square root computed from the symmetric form
    ``S_a^(1/2) S_b S_a^(1/2)``, which has the same eigenvalues.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = a.mean(axis=0) - b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root_a = _sqrt_psd(cov_a)
    cross = np.linalg.eigvalsh(root_a @ cov_b @ root_a)
    trace_root = float(np.sqrt(np.clip(cross, 0.0, None)).sum())
    return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2 * trace_root)


def _distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean distances between every row of ``x`` and every row of ``y``."""
    squared = (x * x).sum(axis=1)[:, None] + (y * y).sum(axis=1)[None, :] - 2 * x @ y.T
    return np.sqrt(np.clip(squared, 0.0, None))


def _inside(query: np.ndarray, points: np.ndarray, k: int, chunk: int = 1024) -> float:
    """Fraction of ``query`` rows within the k-NN radius of at least one row of ``points``.

    Distances are computed ``chunk`` rows at a time, so memory grows with
    ``chunk`` times the set size instead of with its square.
    """
    radii = np.empty(len(points))
    for start in range(0, len(points), chunk):  # column 0 of each sorted row is the point itself
        radii[start:start + chunk] = np.partition(_distances(points[start:start + chunk], points), k, axis=1)[:, k]
    inside = np.zeros(len(query), dtype=bool)
    for start in range(0, len(query), chunk):
        inside[start:start + chunk] = (_distances(query[start:start + chunk], points) <= radii[None, :]).any(axis=1)
    return float(inside.mean())


def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 3) -> tuple[float, float]:
    """Improved Precision & Recall of ``fake`` against ``real`` features.

    Works on 1,024 rows at a time: a few 1,024 x N distance matrices are in
    memory at once (about 0.4 GB each for the 45,000-image full budget).
    """
    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    if min(len(real), len(fake)) <= k:
        raise ValueError(f"each set needs more than k={k} images")
    return _inside(fake, real, k), _inside(real, fake, k)


def reference_sets(manifest: dict, labels: list[int], budget: str = "50") -> tuple[list[int], list[int]]:
    """Real reference images of ``budget`` and a disjoint real set with the same class counts.

    The second set takes, per class, the lowest-index training-pool images
    outside the budget, so it never uses validation images.
    """
    reference = manifest["train_indices"][budget]
    used = set(reference)
    wanted: dict[int, int] = {}
    for index in reference:
        wanted[labels[index]] = wanted.get(labels[index], 0) + 1
    floor = []
    for index in manifest["train_indices"]["full"]:
        label = labels[index]
        if index not in used and wanted.get(label, 0) > 0:
            floor.append(index)
            wanted[label] -= 1
    return reference, sorted(floor)


def load_images(folder: Path) -> np.ndarray:
    """Read every PNG below ``folder`` in path order as (N, 32, 32, 3) uint8."""
    images = []
    for path in sorted(Path(folder).rglob("*.png")):
        image = np.asarray(Image.open(path).convert("RGB"))
        if image.shape != (32, 32, 3):
            raise ValueError(f"{path} is {image.shape[1]}x{image.shape[0]}; generated images must be 32x32")
        images.append(image)
    return np.stack(images)


def image_features(images: np.ndarray, model: Callable, device: torch.device, batch_size: int = 100) -> np.ndarray:
    """Features of (N, 32, 32, 3) uint8 images, prepared as clean-fid's "clean" mode does."""
    from cleanfid.fid import get_batch_features
    from cleanfid.resize import build_resizer

    resize = build_resizer("clean")
    features = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([torch.from_numpy(resize(image).transpose(2, 0, 1).copy())
                             for image in images[start:start + batch_size]])
        features.append(get_batch_features(batch, model, device))
    return np.concatenate(features)


def main() -> None:
    """Score one folder of generated images and the real-versus-real floor, and save both."""
    from importlib.metadata import version

    from cleanfid.features import build_feature_extractor

    from src.component.cifar_splits import labels_sha256
    from src.component.data import load_cifar100, load_manifest
    from src.component.train import LOG_FORMAT, _git_commit

    parser = argparse.ArgumentParser(description="FID and Improved Precision & Recall of generated images.")
    parser.add_argument("--generated", type=Path, required=True, help="folder of generated 32x32 PNGs")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the scores")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--manifest", type=Path, default=Path("src/component/configs/splits/cifar100.json"))
    parser.add_argument("--budget", default="50", help="real reference: the training images of this budget")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    images, labels = load_cifar100(args.data_root, train=True)
    manifest = load_manifest(args.manifest)
    if labels_sha256(labels) != manifest["labels_sha256"]:
        raise ValueError("CIFAR-100 labels do not match the split manifest")
    reference, floor = reference_sets(manifest, labels, args.budget)
    generated = load_images(args.generated)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_extractor("clean", device, use_dataparallel=False)
    real = image_features(images[reference], model, device, args.batch_size)
    scores = {}
    for name, other in (("generated", generated), ("real_floor", images[floor])):
        feats = image_features(other, model, device, args.batch_size)
        precision, recall = precision_recall(real, feats, args.k)
        scores[name] = {"count": len(other), "fid": frechet_distance(real, feats),
                        "precision": precision, "recall": recall}
        logging.info("%s: %d images, FID %.2f, precision %.3f, recall %.3f", name, len(other),
                     scores[name]["fid"], precision, recall)

    result = {
        "generated": str(args.generated),
        "reference": {"budget": args.budget, "count": len(reference)},
        "scores": scores,
        "settings": {"features": "clean-fid clean mode, Inception pool3 (2048)", "resize": "PIL bicubic 32 to 299",
                     "k": args.k, "clean_fid": version("clean-fid")},
        "manifest": str(args.manifest),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_commit(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logging.info("saved %s", args.output)


if __name__ == "__main__":
    main()
