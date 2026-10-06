"""FID and Improved Precision & Recall of a generated pool, at the size of each data budget.

``image_quality`` scores a folder of generated PNGs against the fixed
50-per-class reference. This module scores a ``pool.npz`` (see
``synthetic_pool``) with the same measures and the same features, but at the
size a run actually uses: the first ``--per-class`` images of every class,
against a real reference of the same size.

Reference (training images only; validation and test images are never used):
a class-balanced sample of the 45,000-image training pool with the same number
of images per class, drawn with a fixed seed. When the pool is large enough, a
second, disjoint real sample of the same size gives the real-versus-real
floor: the FID two real samples of that size already have. FID is biased
upwards for small sets, so a score is read against the floor of its size, and
only sets of equal size are compared.

    FID          Frechet distance between Gaussian fits of the two feature sets; lower is closer
    precision    share of generated images inside the real feature manifold (realism)
    recall       share of real images inside the generated feature manifold (variety)
    FID SD       standard deviation of FID over bootstrap resamples of the generated set

    python -m src.component.pool_quality --pool data/synthetic/sd15_prompt/pool.npz \
        --per-class 5 --name sd15_prompt_b5 --generator sd_prompt

Each result goes to ``<pool dir>/quality_<per class>.json`` and one row of
``reports/Latex_report/tables/generator_quality.csv``. With ``--generator``
the scores are also written to that generator's row of the registry.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.component.image_quality import frechet_distance, precision_recall

K_NEIGHBOURS = 3
QUALITY_TABLE = Path("reports/Latex_report/tables/generator_quality.csv")
QUALITY_FIELDS = [
    "name", "pool_sha256", "images", "per_class", "reference", "reference_images", "fid", "fid_sd",
    "fid_real_floor", "precision", "recall", "k", "features", "measured",
]


def bootstrap_fid_sd(real: np.ndarray, fake: np.ndarray, resamples: int, seed: int = 0) -> float | None:
    """Standard deviation of FID over resamples (with replacement) of the generated features."""
    if resamples < 2:
        return None
    rng = np.random.default_rng(seed)
    values = [frechet_distance(real, fake[rng.integers(0, len(fake), len(fake))]) for _ in range(resamples)]
    return float(np.std(values, ddof=1))


def matched_reference(train_labels: np.ndarray, pool_indices: np.ndarray, per_class: int, num_classes: int,
                      seed: int, halves: int = 2) -> list[np.ndarray]:
    """Class-balanced real samples of ``per_class`` images per class, disjoint from each other.

    Returns up to ``halves`` samples (fewer when the pool is too small for
    more), each a sorted array of training-image indices.
    """
    rng = np.random.default_rng(seed)
    pool_indices = np.asarray(pool_indices)
    labels = np.asarray(train_labels)[pool_indices]
    available = min(np.bincount(labels, minlength=num_classes)[:num_classes])
    count = min(halves, available // per_class)
    if count < 1:
        raise ValueError(f"the training pool has fewer than {per_class} images in some class")
    samples = [[] for _ in range(count)]
    for label in range(num_classes):
        chosen = rng.permutation(pool_indices[labels == label])
        for h in range(count):
            samples[h].extend(chosen[h * per_class:(h + 1) * per_class].tolist())
    return [np.sort(np.asarray(s)) for s in samples]


def append_table(path: Path, row: dict) -> None:
    """Add or replace (same name) a row of the quality table."""
    path = Path(path)
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["name"] != row["name"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows + [{k: row.get(k, "") for k in QUALITY_FIELDS}])


def main() -> None:
    """Score one pool at one size against a matched real reference and record the result."""
    import torch
    from cleanfid.features import build_feature_extractor

    from src.component.cifar_splits import labels_sha256
    from src.component.data import budget_indices, load_cifar100, load_manifest
    from src.component.generator_registry import REGISTRY, set_row
    from src.component.image_quality import image_features
    from src.component.synthetic_pool import file_sha256, load_pool, select_per_class

    parser = argparse.ArgumentParser(description="FID and Improved Precision & Recall of a generated pool.")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--name", required=True, help="row name in the quality table, e.g. sd15_prompt_b5")
    parser.add_argument("--per-class", type=int, help="measure the first N images per class (default: all)")
    parser.add_argument("--generator", help="also record the scores in this generator's registry row")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--manifest", type=Path, default=Path("src/component/configs/splits/cifar100.json"))
    parser.add_argument("--bootstrap", type=int, default=10, help="FID resamples for its SD (0 to skip)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the reference sample")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--table", type=Path, default=QUALITY_TABLE)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()

    pool_images, pool_labels, pool_index = load_pool(args.pool)
    per_class = args.per_class or int(np.bincount(pool_labels).min())
    fake_images = pool_images[select_per_class(pool_labels, pool_index, per_class, 100)]

    images, labels = load_cifar100(args.data_root, train=True)
    manifest = load_manifest(args.manifest)
    if labels_sha256(labels) != manifest["labels_sha256"]:
        raise ValueError("CIFAR-100 labels do not match the split manifest")
    references = matched_reference(np.asarray(labels), np.asarray(budget_indices(manifest, "full")),
                                   per_class, 100, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_extractor("clean", device, use_dataparallel=False)
    fake = image_features(fake_images, model, device, args.batch_size).astype(np.float64)
    real = image_features(images[references[0]], model, device, args.batch_size).astype(np.float64)
    fid = frechet_distance(real, fake)
    floor = None
    if len(references) > 1:
        floor = frechet_distance(real, image_features(images[references[1]], model, device, args.batch_size))
    precision, recall = precision_recall(real, fake, K_NEIGHBOURS)
    fid_sd = bootstrap_fid_sd(real, fake, args.bootstrap)

    result = {
        "name": args.name,
        "pool": str(args.pool),
        "pool_sha256": file_sha256(args.pool),
        "images": len(fake),
        "per_class": per_class,
        "reference": f"CIFAR-100 training pool (non-validation), class-balanced, matched count, seed {args.seed}",
        "reference_images": len(real),
        "fid": round(fid, 4),
        "fid_sd": round(fid_sd, 4) if fid_sd is not None else "",
        "fid_real_floor": round(floor, 4) if floor is not None else "",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "k": K_NEIGHBOURS,
        "features": "clean-fid clean mode, Inception pool3 (2048), PIL bicubic 32 to 299",
        "measured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.pool.parent / f"quality_{per_class}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    append_table(args.table, result)
    if args.generator:
        set_row(args.registry, args.generator, {"fid": result["fid"], "precision": result["precision"],
                                                "recall": result["recall"], "quality_file": f"{args.table} ({args.name})"})
    print(f"{args.name}: {len(fake)} images ({per_class}/class) vs {len(real)} real | FID {fid:.2f}"
          + (f" ± {fid_sd:.2f}" if fid_sd is not None else "")
          + (f" (real-vs-real floor {floor:.2f})" if floor is not None else "")
          + f" | precision {precision:.3f} | recall {recall:.3f}")


if __name__ == "__main__":
    main()
