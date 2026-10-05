"""Tests for the real-versus-generated sample figure."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.component.sample_grid import grid_rows, plot_grid


class SampleGridTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.real = rng.integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)
        self.real_labels = [i % 2 for i in range(20)]
        self.pool = rng.integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
        self.pool_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.pool_index = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    def test_rows_take_budget_images_and_first_generated_images(self) -> None:
        rows = grid_rows(self.real, self.real_labels, [0, 1, 2, 3, 4, 5], self.pool, self.pool_labels,
                         self.pool_index, classes=[1], per_side=2)
        label, real, synthetic = rows[0]
        self.assertEqual(label, 1)
        self.assertTrue(np.array_equal(real[0], self.real[1]) and np.array_equal(real[1], self.real[3]))
        self.assertTrue(np.array_equal(synthetic[0], self.pool[4]) and len(synthetic) == 2)

    def test_writes_svg_pdf_and_png(self) -> None:
        rows = grid_rows(self.real, self.real_labels, list(range(20)), self.pool, self.pool_labels,
                         self.pool_index, classes=[0, 1], per_side=3)
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_grid(rows, ["apple", "aquarium_fish"], Path(tmp) / "grid", "generated")
            self.assertEqual(sorted(p.suffix for p in paths), [".pdf", ".png", ".svg"])
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))


if __name__ == "__main__":
    unittest.main()
