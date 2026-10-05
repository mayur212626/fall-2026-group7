"""Figure of real and generated CIFAR-100 images side by side.

Each row is one class: real training images from the smallest budget on the
left, the first generated images of the pool on the right, both at the 32x32
resolution the classifiers see. Used for the report and the presentation.

    python -m src.component.sample_grid --pool data/synthetic/sd15_prompt/pool.npz \
        --output reports/Latex_report/fig/sd15_prompt_samples
"""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from src.component.synthetic_pool import load_pool

DEFAULT_CLASSES = [0, 3, 8, 19, 30, 43, 58, 69, 82, 97]  # apple, bear, bicycle, cattle, dolphin, lion, pickup_truck, rocket, sunflower, wolf


def grid_rows(
    real_images: np.ndarray, real_labels: Sequence[int], real_indices: Sequence[int],
    pool_images: np.ndarray, pool_labels: np.ndarray, pool_index: np.ndarray,
    classes: Sequence[int], per_side: int,
) -> list[tuple[int, list[np.ndarray], list[np.ndarray]]]:
    """Return ``(class, real images, generated images)`` for every class in ``classes``."""
    real_labels = np.asarray(real_labels)
    rows = []
    for label in classes:
        real = [real_images[i] for i in real_indices if real_labels[i] == label][:per_side]
        mask = (pool_labels == label) & (pool_index < per_side)
        synthetic = [pool_images[i] for i in np.flatnonzero(mask)]
        rows.append((label, real, synthetic))
    return rows


def plot_grid(rows: list, class_names: Sequence[str], out_stem: Path, generator_label: str) -> list[Path]:
    """Draw the rows and save ``out_stem`` as SVG, PDF and PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
    per_side = max(len(r[1]) for r in rows)
    cols = 2 * per_side + 1  # real | gap | generated
    fig, axes = plt.subplots(len(rows), cols, figsize=(0.62 * cols + 1.2, 0.66 * len(rows) + 0.5), squeeze=False)
    for r, (label, real, synthetic) in enumerate(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if c < per_side and c < len(real):
                ax.imshow(real[c], interpolation="nearest")
            elif c > per_side and c - per_side - 1 < len(synthetic):
                ax.imshow(synthetic[c - per_side - 1], interpolation="nearest")
        axes[r][0].set_ylabel(class_names[label].replace("_", " "), rotation=0, ha="right", va="center")
    axes[0][per_side // 2].set_title("Real (CIFAR-100)", loc="center")
    axes[0][per_side + 1 + per_side // 2].set_title(generator_label, loc="center")
    fig.subplots_adjust(wspace=0.05, hspace=0.08, left=0.16, right=0.99, top=0.93, bottom=0.01)
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_stem.with_suffix(ext) for ext in (".svg", ".pdf", ".png")]
    for path in paths:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Real vs generated CIFAR-100 sample grid.")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="path without extension")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--manifest", type=Path, default=Path("src/component/configs/splits/cifar100.json"))
    parser.add_argument("--classes", type=int, nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--per-side", type=int, default=5)
    parser.add_argument("--label", default="Stable Diffusion 1.5, class prompts")
    args = parser.parse_args()

    from torchvision.datasets import CIFAR100

    from src.component.data import budget_indices, load_manifest

    dataset = CIFAR100(args.data_root, train=True, download=False)
    manifest = load_manifest(args.manifest)
    pool_images, pool_labels, pool_index = load_pool(args.pool)
    rows = grid_rows(dataset.data, dataset.targets, budget_indices(manifest, "5"),
                     pool_images, pool_labels, pool_index, args.classes, args.per_side)
    for path in plot_grid(rows, dataset.classes, args.output, args.label):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
