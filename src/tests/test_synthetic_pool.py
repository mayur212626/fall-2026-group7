"""Tests for generated image pools (NumPy only)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.component.synthetic_pool import file_sha256, load_pool, save_pool, select_per_class


def fake_pool(num_classes: int, per_class: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Images whose first pixel encodes (class, index), in shuffled order."""
    labels = np.repeat(np.arange(num_classes), per_class)
    index = np.tile(np.arange(per_class), num_classes)
    images = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = labels
    images[:, 0, 0, 1] = index
    order = np.random.default_rng(0).permutation(len(labels))
    return images[order], labels[order], index[order]


class SavePoolTest(unittest.TestCase):
    def test_round_trip_sorted_by_class_then_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.npz"
            sha = save_pool(path, *fake_pool(3, 4))
            images, labels, index = load_pool(path)
        self.assertEqual(labels.tolist(), [0] * 4 + [1] * 4 + [2] * 4)
        self.assertEqual(index.tolist(), [0, 1, 2, 3] * 3)
        self.assertTrue(np.array_equal(images[:, 0, 0, 0], labels))
        self.assertTrue(np.array_equal(images[:, 0, 0, 1], index))
        self.assertEqual(len(sha), 64)

    def test_same_content_gives_same_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = save_pool(Path(tmp) / "a.npz", *fake_pool(2, 3))
            b = save_pool(Path(tmp) / "b.npz", *fake_pool(2, 3))
            self.assertEqual(a, b)
            self.assertEqual(a, file_sha256(Path(tmp) / "a.npz"))

    def test_rejects_gaps_in_index(self) -> None:
        images, labels, index = fake_pool(2, 3)
        index = np.where(index == 2, 5, index)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            save_pool(Path(tmp) / "p.npz", images, labels, index)

    def test_rejects_non_uint8_images(self) -> None:
        images, labels, index = fake_pool(2, 3)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            save_pool(Path(tmp) / "p.npz", images.astype(np.float32), labels, index)


class SelectPerClassTest(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_pool(Path(tmp) / "p.npz", *fake_pool(3, 6))
            _, self.labels, self.index = load_pool(Path(tmp) / "p.npz")

    def test_takes_the_first_k_images_of_every_class(self) -> None:
        rows = select_per_class(self.labels, self.index, 2, 3)
        self.assertEqual(np.bincount(self.labels[rows]).tolist(), [2, 2, 2])
        self.assertTrue((self.index[rows] < 2).all())

    def test_selections_are_nested(self) -> None:
        small = set(select_per_class(self.labels, self.index, 2, 3).tolist())
        large = set(select_per_class(self.labels, self.index, 5, 3).tolist())
        self.assertLess(small, large)

    def test_rejects_a_pool_that_is_too_small(self) -> None:
        with self.assertRaises(ValueError):
            select_per_class(self.labels, self.index, 7, 3)

    def test_rejects_a_missing_class(self) -> None:
        with self.assertRaises(ValueError):
            select_per_class(self.labels, self.index, 1, 4)


if __name__ == "__main__":
    unittest.main()
