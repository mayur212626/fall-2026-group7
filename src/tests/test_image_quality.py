"""Tests for the generated-image quality measures. No feature network is downloaded."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm

from src.component.image_quality import (
    frechet_distance,
    image_features,
    load_images,
    precision_recall,
    reference_sets,
)


def features(count: int = 200, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Return random features with a non-zero mean and correlated dimensions."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, dim)) @ rng.normal(size=(dim, dim)) + rng.normal(size=dim)


class FrechetDistanceTest(unittest.TestCase):
    def test_identical_sets_have_zero_distance(self) -> None:
        a = features()
        self.assertAlmostEqual(frechet_distance(a, a), 0.0, places=6)

    def test_a_shift_adds_its_squared_length(self) -> None:
        a = features()
        shift = np.arange(8, dtype=float)
        self.assertAlmostEqual(frechet_distance(a, a + shift), float(shift @ shift), places=6)

    def test_doubling_matches_the_closed_form(self) -> None:
        a = features()
        mean = a.mean(axis=0)
        expected = float(mean @ mean) + float(np.trace(np.cov(a, rowvar=False)))
        self.assertAlmostEqual(frechet_distance(a, 2 * a), expected, places=6)

    def test_is_symmetric(self) -> None:
        a, b = features(seed=0), features(seed=1)
        self.assertAlmostEqual(frechet_distance(a, b), frechet_distance(b, a), places=6)

    def test_matches_the_textbook_formula_with_a_general_matrix_square_root(self) -> None:
        a, b = features(seed=0), features(seed=1) * 2 + 1
        cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
        diff = a.mean(axis=0) - b.mean(axis=0)
        expected = diff @ diff + np.trace(cov_a + cov_b - 2 * sqrtm(cov_a @ cov_b).real)
        self.assertAlmostEqual(frechet_distance(a, b), float(expected), places=4)


class PrecisionRecallTest(unittest.TestCase):
    def test_identical_sets_score_one(self) -> None:
        a = features()
        self.assertEqual(precision_recall(a, a, k=3), (1.0, 1.0))

    def test_distant_sets_score_zero(self) -> None:
        a = features()
        self.assertEqual(precision_recall(a, a + 1000.0, k=3), (0.0, 0.0))

    def test_radius_is_the_distance_to_the_kth_neighbour(self) -> None:
        real = np.array([[0.0], [1.0], [3.0]])  # 1st-neighbour radii 1, 1, 2; 2nd-neighbour radii 3, 2, 3
        self.assertEqual(precision_recall(real, np.full((3, 1), 1.5), k=1)[0], 1.0)
        self.assertEqual(precision_recall(real, np.full((3, 1), 5.5), k=1)[0], 0.0)
        self.assertEqual(precision_recall(real, np.full((3, 1), 5.5), k=2)[0], 1.0)

    def test_rejects_sets_with_k_or_fewer_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "more than k=3"):
            precision_recall(features(count=10), features(count=3), k=3)

    def test_collapse_onto_one_real_image_keeps_precision_and_loses_recall(self) -> None:
        real = features(count=50)
        fake = np.repeat(real[:1], 50, axis=0)
        precision, recall = precision_recall(real, fake, k=3)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1 / 50)


class ReferenceSetsTest(unittest.TestCase):
    """Two classes of 8 training-pool images each; the 2-per-class budget is the reference."""

    labels = [i % 2 for i in range(20)]
    manifest = {
        "validation_indices": [0, 1, 2, 3],
        "train_indices": {"2": [4, 5, 6, 7], "full": list(range(4, 20))},
    }

    def test_reference_is_the_budget_and_the_floor_is_disjoint_with_the_same_classes(self) -> None:
        reference, floor = reference_sets(self.manifest, self.labels, budget="2")
        self.assertEqual(reference, [4, 5, 6, 7])
        self.assertEqual(floor, [8, 9, 10, 11])

    def test_floor_never_uses_validation_images(self) -> None:
        _, floor = reference_sets(self.manifest, self.labels, budget="2")
        self.assertFalse(set(floor) & set(self.manifest["validation_indices"]))


class LoadImagesTest(unittest.TestCase):
    def write(self, folder: Path, name: str, size: int, value: int) -> None:
        Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8)).save(folder / name)

    def test_reads_32_pixel_pngs_in_path_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "c1").mkdir()
            self.write(folder / "c1", "0.png", 32, 20)
            self.write(folder, "a.png", 32, 10)
            images = load_images(folder)
        self.assertEqual(images.shape, (2, 32, 32, 3))
        self.assertEqual(images.dtype, np.uint8)
        self.assertEqual([int(images[0, 0, 0, 0]), int(images[1, 0, 0, 0])], [10, 20])

    def test_rejects_images_that_were_not_resized_to_32_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), "big.png", 512, 0)
            with self.assertRaisesRegex(ValueError, "32x32"):
                load_images(Path(tmp))


class ImageFeaturesTest(unittest.TestCase):
    def test_feeds_299_pixel_images_on_the_0_255_scale_in_batches(self) -> None:
        seen = []

        def model(batch: torch.Tensor) -> torch.Tensor:
            seen.append(tuple(batch.shape))
            return batch.mean(dim=(2, 3))

        images = np.full((5, 32, 32, 3), 200, dtype=np.uint8)
        out = image_features(images, model, torch.device("cpu"), batch_size=2)
        self.assertEqual(seen, [(2, 3, 299, 299), (2, 3, 299, 299), (1, 3, 299, 299)])
        self.assertEqual(out.shape, (5, 3))
        np.testing.assert_allclose(out, 200.0, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
