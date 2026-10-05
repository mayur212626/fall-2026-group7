"""Generated image pools: file format, nested selection and fingerprint.

A pool is one ``.npz`` file with every generated image already reduced to the
real images' resolution (32x32 for CIFAR-100):

    images          (N, 32, 32, 3) uint8
    labels          (N,) int64, class of each image
    index_in_class  (N,) int64, 0, 1, 2, ... within each class, in generation order

Rows are sorted by class and then by ``index_in_class``. A run that needs
``k`` synthetic images per class takes the images with ``index_in_class < k``,
so the images of a smaller budget are always contained in those of a larger
one, like the real training budgets.

This module needs only NumPy, so the pool can be checked without PyTorch.
"""

import hashlib
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_pool(path: Path, images: np.ndarray, labels: np.ndarray, index_in_class: np.ndarray) -> str:
    """Validate, sort by (class, index) and save a pool; return its SHA-256.

    Raises:
        ValueError: If the arrays disagree in length, the images are not
            (N, H, W, 3) uint8, or a class has a repeated or missing index.
    """
    images = np.asarray(images)
    labels = np.asarray(labels, dtype=np.int64)
    index_in_class = np.asarray(index_in_class, dtype=np.int64)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"images must be (N, H, W, 3) uint8, got {images.shape} {images.dtype}")
    if not len(images) == len(labels) == len(index_in_class):
        raise ValueError("images, labels and index_in_class must have the same length")
    order = np.lexsort((index_in_class, labels))
    images, labels, index_in_class = images[order], labels[order], index_in_class[order]
    for label in np.unique(labels):
        found = index_in_class[labels == label]
        if not np.array_equal(found, np.arange(len(found))):
            raise ValueError(f"class {label}: index_in_class must be 0..{len(found) - 1} without gaps")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, images=images, labels=labels, index_in_class=index_in_class)
    tmp.replace(path)
    return file_sha256(path)


def load_pool(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(images, labels, index_in_class)`` from a pool file."""
    with np.load(Path(path)) as data:
        return data["images"], data["labels"].astype(np.int64), data["index_in_class"].astype(np.int64)


def select_per_class(labels: np.ndarray, index_in_class: np.ndarray, per_class: int, num_classes: int) -> np.ndarray:
    """Return the pool rows of the first ``per_class`` images of every class.

    Raises:
        ValueError: If any of the ``num_classes`` classes has fewer than
            ``per_class`` images in the pool.
    """
    labels = np.asarray(labels)
    index_in_class = np.asarray(index_in_class)
    counts = np.bincount(labels, minlength=num_classes)
    short = np.flatnonzero(counts[:num_classes] < per_class)
    if len(short):
        raise ValueError(
            f"pool has fewer than {per_class} images for {len(short)} classes (e.g. class {short[0]}: "
            f"{counts[short[0]]})"
        )
    return np.flatnonzero(index_in_class < per_class)
