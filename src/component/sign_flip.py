"""Diagnostics for the sign flip: why SD images help pretrained classifiers but not randomly initialized ones.

Stage 1 found that Stable Diffusion class-prompt images raise the accuracy of
ImageNet-pretrained classifiers at every budget but lose to RandAugment for
randomly initialized ones. Two explanations are tested here, class by class,
against the per-class recall changes written by ``error_analysis``:

- diversity: a scratch network overfits to the few modes a prompt produces.
  If so, the scratch gain of a class should rise with how much of the real
  class the generated images cover (Improved Recall of that class), more
  than the pretrained gain does;
- domain gap: generated images lie off the CIFAR distribution and only a
  pretrained backbone bridges the gap. If so, a classifier trained on real
  images only should recognise generated images much less well than real
  validation images, more so from scratch, and the scratch gain of a class
  should fall with the distance between its generated and real images
  (precision, centroid gap).

The same per-class measures also test RQ3 at the level of classes: which
property of the generated images predicts how much they help.

Only training images, the generated pool and validation predictions are used;
the test split is never read. Four steps, the first two on the machine with
the pool and a GPU:

    python -m src.component.sign_flip features --pool data/synthetic/sd15_prompt/pool.npz \
        --output data/synthetic/sd15_prompt/class_features_50.npz
    python -m src.component.sign_flip class-quality --features data/synthetic/sd15_prompt/class_features_50.npz
    python -m src.component.sign_flip synthetic-accuracy --pool data/synthetic/sd15_prompt/pool.npz \
        --runs runs/stage1-pretrained/pretrained_resnet50_real_only_bfull_s0 ...
    python -m src.component.sign_flip correlate

``synthetic-accuracy`` needs the ``best.pt`` of real-only runs, so it runs where
those checkpoints are. ``correlate`` reads only CSV files.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

NUM_CLASSES = 100
K_NEIGHBOURS = 3
PERMUTATIONS = 10_000
PERMUTATION_SEED = 0
REFERENCE_SEED = 0
TABLES = Path("reports/Latex_report/tables")
FIGURES = Path("reports/Latex_report/fig")
QUALITY_CSV = TABLES / "sign_flip_class_quality.csv"
SYNTHETIC_ACCURACY_CSV = TABLES / "sign_flip_synthetic_accuracy.csv"
# Per-class recall changes from error_analysis (budget 5) that the measures are tested against.
CHANGE_FILES = {
    ("pretrained", "sd_prompt", "real_only"): "stage1_pretrained_b5_sd_prompt_vs_real_only_classes.csv",
    ("pretrained", "sd_prompt", "randaugment"): "stage1_pretrained_b5_sd_prompt_vs_randaugment_classes.csv",
    ("scratch", "sd_prompt", "real_only"): "stage1_scratch_b5_sd_prompt_vs_real_only_classes.csv",
    ("scratch", "sd_prompt", "randaugment"): "stage1_scratch_b5_sd_prompt_vs_randaugment_classes.csv",
}
# Per-class measures and what a high value means.
MEASURES = {
    "recall": "share of the real class covered by its generated images (diversity)",
    "precision": "share of generated images inside the real class (realism)",
    "diversity_ratio": "spread of generated images relative to the real class",
    "centroid_gap": "distance between generated and real class centres (domain gap)",
}
MODEL_NAMES = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16"}
POINT_COLOR = "#2a78d6"


# --------------------------------------------------------------------------- per-class feature measures
def _distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    squared = (x * x).sum(axis=1)[:, None] + (y * y).sum(axis=1)[None, :] - 2 * x @ y.T
    return np.sqrt(np.clip(squared, 0.0, None))


def _inside(query: np.ndarray, points: np.ndarray, k: int) -> float:
    """Share of ``query`` rows within the k-NN radius of at least one row of ``points``."""
    radii = np.partition(_distances(points, points), k, axis=1)[:, k]  # column 0 is the point itself
    return float((_distances(query, points) <= radii[None, :]).any(axis=1).mean())


def _mean_pairwise(x: np.ndarray) -> float:
    d = _distances(x, x)
    return float(d[np.triu_indices(len(x), 1)].mean())


def class_quality(real: np.ndarray, fake: np.ndarray, real_labels: np.ndarray, fake_labels: np.ndarray,
                  k: int = K_NEIGHBOURS) -> list[dict]:
    """Improved Precision & Recall, diversity ratio and centroid gap of the generated images of every class.

    Each class is compared with real images of the same class only:
    precision is the share of its generated images inside the real class
    manifold, recall the share of its real images inside the generated one
    (k-NN balls, Kynkaanniemi et al.), diversity ratio the mean pairwise
    distance of the generated images over that of the real ones, and
    centroid gap the distance between the two class means over the mean
    pairwise real distance.
    """
    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    rows = []
    for label in range(int(max(real_labels.max(), fake_labels.max())) + 1):
        r, f = real[real_labels == label], fake[fake_labels == label]
        if min(len(r), len(f)) <= k:
            raise ValueError(f"class {label} needs more than k={k} real and generated images")
        spread = _mean_pairwise(r)
        rows.append({
            "class": label,
            "real_images": len(r),
            "generated_images": len(f),
            "precision": _inside(f, r, k),
            "recall": _inside(r, f, k),
            "diversity_ratio": _mean_pairwise(f) / spread,
            "centroid_gap": float(np.linalg.norm(r.mean(axis=0) - f.mean(axis=0))) / spread,
        })
    return rows


def reference_indices(train_labels: np.ndarray, pool_indices: np.ndarray, per_class: int,
                      num_classes: int = NUM_CLASSES, seed: int = REFERENCE_SEED) -> np.ndarray:
    """``per_class`` training images of every class, drawn with a fixed seed from the non-validation pool."""
    rng = np.random.default_rng(seed)
    pool_indices = np.asarray(pool_indices)
    labels = np.asarray(train_labels)[pool_indices]
    chosen = []
    for label in range(num_classes):
        members = pool_indices[labels == label]
        if len(members) < per_class:
            raise ValueError(f"class {label} has only {len(members)} training images")
        chosen.extend(rng.choice(members, per_class, replace=False).tolist())
    return np.sort(np.asarray(chosen))


# --------------------------------------------------------------------------- correlation
def rank(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties given their average rank."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for v in np.unique(values):
        tied = values == v
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation of ``x`` and ``y``; NaN when either is constant."""
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denominator) if denominator else float("nan")


def permutation_p(x: np.ndarray, y: np.ndarray, permutations: int = PERMUTATIONS,
                  seed: int = PERMUTATION_SEED) -> float:
    """Two-sided permutation p-value of the Spearman correlation (y shuffled), with the +1 correction."""
    observed = abs(spearman(x, y))
    if np.isnan(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    rx = rank(x) - (len(x) + 1) / 2
    ry = rank(y) - (len(y) + 1) / 2
    norm = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    hits = 0
    for start in range(0, permutations, 1000):
        n = min(1000, permutations - start)
        shuffled = np.stack([rng.permutation(ry) for _ in range(n)])
        hits += int((np.abs(shuffled @ rx / norm) >= observed - 1e-12).sum())
    return (hits + 1) / (permutations + 1)


# --------------------------------------------------------------------------- files
def read_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict], path: Path) -> Path:
    def cell(value: object) -> object:
        return round(value, 4) if isinstance(value, float) else value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: cell(v) for k, v in row.items()} for row in rows)
    return path


def read_changes(path: Path) -> dict[str, np.ndarray]:
    """Per-class recall change of each model from an error_analysis ``*_classes.csv``, indexed by class."""
    changes: dict[str, np.ndarray] = {}
    for row in read_rows(path):
        changes.setdefault(row["model"], np.full(NUM_CLASSES, np.nan))[int(row["class"])] = float(row["recall_change"])
    return changes


def add_randaugment_changes(changes: dict[tuple[str, str, str], dict[str, np.ndarray]]) -> None:
    """Add the per-class change of RandAugment over real-only, derived from SD - real and SD - RandAugment.

    The SD - RandAugment change mixes what SD adds and what RandAugment adds,
    so a measure that correlates with it may only track RandAugment. The
    derived (SD - real) - (SD - RA) = RA - real separates the two; it uses the
    seeds both tables share, so it is exact for those seeds.
    """
    for init in sorted({key[0] for key in changes}):
        sd_real, sd_ra = changes.get((init, "sd_prompt", "real_only")), changes.get((init, "sd_prompt", "randaugment"))
        if sd_real and sd_ra:
            changes[(init, "randaugment", "real_only")] = {m: sd_real[m] - sd_ra[m] for m in sd_real if m in sd_ra}


def synthetic_accuracy_measures(rows: list[dict]) -> dict[tuple[str, str], np.ndarray]:
    """Per-class accuracy on generated images, averaged over the runs of each (init, model)."""
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for row in rows:
        grouped.setdefault((row["init"], row["model"]), []).append(
            np.array([float(v) for v in row["synthetic_class_accuracy"].split()]))
    return {key: np.mean(values, axis=0) for key, values in grouped.items()}


def correlation_rows(quality: list[dict], changes: dict[tuple[str, str, str], dict[str, np.ndarray]],
                     synthetic: dict[tuple[str, str], np.ndarray] | None = None,
                     permutations: int = PERMUTATIONS) -> list[dict]:
    """Spearman correlation of every per-class measure with every per-class recall change."""
    by_class = sorted(quality, key=lambda r: int(r["class"]))
    measures = {m: np.array([float(r[m]) for r in by_class]) for m in MEASURES}
    rows = []
    for (init, treatment, control), per_model in changes.items():
        for model, change in per_model.items():
            available = dict(measures)
            if synthetic and (init, model) in synthetic:
                available["synthetic_accuracy"] = synthetic[(init, model)]
            for name, values in available.items():
                keep = ~np.isnan(change) & ~np.isnan(values)
                rows.append({"init": init, "model": model, "treatment": treatment, "control": control,
                             "measure": name, "classes": int(keep.sum()),
                             "spearman": spearman(values[keep], change[keep]),
                             "p_value": permutation_p(values[keep], change[keep], permutations)})
    return rows


def correlation_table(rows: list[dict]) -> str:
    """Booktabs table of the correlations: one row per measure, one column per (init, model, comparison)."""
    columns = sorted({(r["init"], r["model"], r["treatment"], r["control"]) for r in rows},
                     key=lambda c: (c[0], c[3], list(MODEL_NAMES).index(c[1])))
    measures = [m for m in [*MEASURES, "synthetic_accuracy"] if any(r["measure"] == m for r in rows)]
    cell = {(r["init"], r["model"], r["treatment"], r["control"], r["measure"]): r for r in rows}
    short = {"real_only": "real", "randaugment": "RA", "sd_prompt": "SD"}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Sign-flip diagnostics at 5 images per class: Spearman correlation over the 100 classes between "
        r"a per-class property of the SD class-prompt images and the per-class recall change of SD over its "
        r"control (two-sided permutation $p$ in brackets). RA$-$real is derived as (SD$-$real) $-$ (SD$-$RA) and "
        r"shows how much of an SD$-$RA correlation comes from RandAugment alone. Diversity predicts a positive correlation with recall "
        r"and diversity ratio, a domain gap a negative one with centroid gap and a positive one with precision "
        r"and with the accuracy of a real-only classifier on the generated images.}",
        r"\label{tab:sign-flip}",
        r"\begin{tabular}{l" + "r" * len(columns) + "}",
        r"\toprule",
        "Measure & " + " & ".join(f"{i[:4]}. {MODEL_NAMES[m]} {short[t]}$-${short[c]}" for i, m, t, c in columns) + r" \\",
        r"\midrule",
    ]
    for measure in measures:
        values = []
        for i, m, t, c in columns:
            r = cell.get((i, m, t, c, measure))
            values.append("--" if r is None or np.isnan(float(r["spearman"]))
                          else f"{float(r['spearman']):+.2f} ({float(r['p_value']):.3f})")
        lines.append(measure.replace("_", " ") + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def plot_recall_vs_change(quality: list[dict], changes: dict[tuple[str, str, str], dict[str, np.ndarray]],
                          comparison: tuple[str, str], out_stem: Path, measure: str = "recall") -> list[Path]:
    """Scatter of a per-class measure against the per-class recall change, one panel per init and model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_class = sorted(quality, key=lambda r: int(r["class"]))
    x = np.array([float(r[measure]) for r in by_class])
    panels = [(init, model, per_model[model]) for (init, t, c), per_model in sorted(changes.items())
              if (t, c) == comparison for model in MODEL_NAMES if model in per_model]
    if not panels:
        return []
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(1, len(panels), figsize=(2.8 * len(panels), 2.8), squeeze=False)
    for ax, (init, model, y) in zip(axes[0], panels, strict=True):
        keep = ~np.isnan(y)
        ax.scatter(x[keep], y[keep], s=12, color=POINT_COLOR, alpha=0.7, linewidths=0)
        ax.axhline(0, color="#8a8a86", linewidth=0.8)
        ax.set_title(f"{init}, {MODEL_NAMES[model]}\n$\\rho$ = {spearman(x[keep], y[keep]):+.2f}", fontsize=9)
        ax.set_xlabel(f"Per-class {measure.replace('_', ' ')}")
        ax.grid(True, color="#e0e0e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("Recall change (pp)")
    fig.tight_layout()
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_stem.with_suffix(".svg"), out_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- commands needing torch
def compute_features(pool: Path, per_class: int, data_root: Path, manifest_path: Path, output: Path,
                     batch_size: int) -> Path:
    """Inception features (clean-fid "clean" mode) of ``per_class`` real and generated images of every class."""
    import torch
    from cleanfid.features import build_feature_extractor

    from src.component.cifar_splits import labels_sha256
    from src.component.data import budget_indices, load_cifar100, load_manifest
    from src.component.image_quality import image_features
    from src.component.synthetic_pool import load_pool, select_per_class

    images, labels = load_cifar100(data_root, train=True)
    manifest = load_manifest(manifest_path)
    if labels_sha256(labels) != manifest["labels_sha256"]:
        raise ValueError("CIFAR-100 labels do not match the split manifest")
    real_rows = reference_indices(np.asarray(labels), np.asarray(budget_indices(manifest, "full")), per_class)
    pool_images, pool_labels, pool_index = load_pool(pool)
    fake_rows = select_per_class(pool_labels, pool_index, per_class, NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_extractor("clean", device, use_dataparallel=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        real=image_features(images[real_rows], model, device, batch_size).astype(np.float32),
        real_labels=np.asarray(labels)[real_rows],
        real_indices=real_rows,
        fake=image_features(pool_images[fake_rows], model, device, batch_size).astype(np.float32),
        fake_labels=pool_labels[fake_rows],
        fake_rows=fake_rows,
    )
    return output


def synthetic_accuracy(run_dirs: list[Path], pool: Path, per_class: int, batch_size: int) -> list[dict]:
    """Accuracy of each run's best checkpoint on generated images, next to its validation accuracy."""
    import torch

    from src.component.data import to_model_input
    from src.component.models import build_model
    from src.component.synthetic_pool import load_pool, select_per_class

    pool_images, pool_labels, pool_index = load_pool(pool)
    rows_used = select_per_class(pool_labels, pool_index, per_class, NUM_CLASSES)
    images, labels = pool_images[rows_used], pool_labels[rows_used]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run_dir in run_dirs:
        settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"]
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        model = build_model(settings["model"], False, settings["num_classes"])
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["model"])
        model.to(device).eval()
        predicted = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = torch.from_numpy(images[start:start + batch_size]).permute(0, 3, 1, 2).to(device)
                predicted.append(model(to_model_input(batch, settings["image_size"])).argmax(1).cpu().numpy())
        predicted = np.concatenate(predicted)
        correct = np.bincount(labels[predicted == labels], minlength=NUM_CLASSES)
        per_class_accuracy = 100.0 * correct / np.bincount(labels, minlength=NUM_CLASSES)
        rows.append({
            "run": run_dir.name, "init": settings["init"], "model": settings["model"],
            "condition": settings["condition"], "budget": settings["budget"], "seed": settings["seed"],
            "validation_accuracy": float(result["best_validation"]["accuracy"]),
            "synthetic_accuracy": 100.0 * float((predicted == labels).mean()),
            "generated_images": len(labels),
            "synthetic_class_accuracy": " ".join(f"{v:.2f}" for v in per_class_accuracy),
        })
        print(f"{run_dir.name}: validation {rows[-1]['validation_accuracy']:.2f}%, "
              f"generated {rows[-1]['synthetic_accuracy']:.2f}%")
    return rows


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    feats = commands.add_parser("features", help="Inception features of real and generated images per class")
    feats.add_argument("--pool", type=Path, default=Path("data/synthetic/sd15_prompt/pool.npz"))
    feats.add_argument("--per-class", type=int, default=50)
    feats.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    feats.add_argument("--manifest", type=Path, default=Path("src/component/configs/splits/cifar100.json"))
    feats.add_argument("--output", type=Path, default=Path("data/synthetic/sd15_prompt/class_features_50.npz"))
    feats.add_argument("--batch-size", type=int, default=100)

    quality = commands.add_parser("class-quality", help="per-class precision, recall, diversity and gap")
    quality.add_argument("--features", type=Path, default=Path("data/synthetic/sd15_prompt/class_features_50.npz"))
    quality.add_argument("--output", type=Path, default=QUALITY_CSV)

    acc = commands.add_parser("synthetic-accuracy", help="accuracy of real-only checkpoints on generated images")
    acc.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders with best.pt")
    acc.add_argument("--pool", type=Path, default=Path("data/synthetic/sd15_prompt/pool.npz"))
    acc.add_argument("--per-class", type=int, default=50)
    acc.add_argument("--batch-size", type=int, default=250)
    acc.add_argument("--output", type=Path, default=SYNTHETIC_ACCURACY_CSV)

    corr = commands.add_parser("correlate", help="correlate the per-class measures with the per-class gains")
    corr.add_argument("--quality", type=Path, default=QUALITY_CSV)
    corr.add_argument("--synthetic-accuracy", type=Path, default=SYNTHETIC_ACCURACY_CSV)
    corr.add_argument("--tables", type=Path, default=TABLES)
    corr.add_argument("--figures", type=Path, default=FIGURES)
    corr.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()

    if args.command == "features":
        path = compute_features(args.pool, args.per_class, args.data_root, args.manifest, args.output, args.batch_size)
        print(f"wrote {path}")
    elif args.command == "class-quality":
        with np.load(args.features) as f:
            rows = class_quality(f["real"], f["fake"], f["real_labels"], f["fake_labels"])
        print(f"wrote {write_rows(rows, args.output)}")
        for m in MEASURES:
            values = np.array([r[m] for r in rows])
            print(f"  {m}: mean {values.mean():.3f}, range {values.min():.3f} to {values.max():.3f}")
    elif args.command == "synthetic-accuracy":
        rows = synthetic_accuracy(args.runs, args.pool, args.per_class, args.batch_size)
        print(f"wrote {write_rows(rows, args.output)}")
    else:
        quality_rows = read_rows(args.quality)
        changes = {key: read_changes(args.tables / name) for key, name in CHANGE_FILES.items()
                   if (args.tables / name).exists()}
        add_randaugment_changes(changes)
        synthetic = (synthetic_accuracy_measures(read_rows(args.synthetic_accuracy))
                     if args.synthetic_accuracy.exists() else None)
        rows = correlation_rows(quality_rows, changes, synthetic, args.permutations)
        write_rows(rows, args.tables / "sign_flip_correlations.csv")
        (args.tables / "sign_flip_correlations.tex").write_text(correlation_table(rows), encoding="utf-8")
        for comparison in (("sd_prompt", "real_only"), ("sd_prompt", "randaugment")):
            for measure in ("recall", "centroid_gap"):
                stem = args.figures / f"sign_flip_{measure}_{comparison[0]}_vs_{comparison[1]}"
                for path in plot_recall_vs_change(quality_rows, changes, comparison, stem, measure):
                    print(f"wrote {path}")
        print("Spearman correlation of per-class measure with per-class recall change (b5):")
        for r in rows:
            name = f"{'RA' if r['treatment'] == 'randaugment' else 'SD'}-{r['control']}"
            print(f"  {r['init']:<10} {MODEL_NAMES[r['model']]:<9} {name:<14} {r['measure']:<18} "
                  f"rho {r['spearman']:+.2f}  p {r['p_value']:.4f}  (n={r['classes']})")


if __name__ == "__main__":
    main()
