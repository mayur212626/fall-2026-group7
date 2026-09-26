"""Tests for the CIFAR-100 data pipeline."""

import unittest

import numpy as np
import torch

from src.component.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CifarImages,
    StepBatchSampler,
    budget_indices,
    to_model_input,
)


def fake_images(count: int) -> np.ndarray:
    """Return distinct random 32x32 RGB uint8 images, shaped like CIFAR's data."""
    return np.random.default_rng(0).integers(0, 256, (count, 32, 32, 3), dtype=np.uint8)


def stream(sampler: StepBatchSampler) -> list[tuple[int, int]]:
    """Flatten a sampler's batches into one list of (position, visit seed)."""
    return [item for batch in sampler for item in batch]


class BudgetIndicesTest(unittest.TestCase):
    manifest = {"train_indices": {"5": [3, 8], "full": [1, 3, 8]}}

    def test_returns_manifest_indices(self) -> None:
        self.assertEqual(budget_indices(self.manifest, "5"), [3, 8])

    def test_rejects_unknown_budget(self) -> None:
        with self.assertRaises(ValueError):
            budget_indices(self.manifest, "7")


class CifarImagesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.images = fake_images(10)
        self.labels = list(range(10))

    def test_returns_original_pixels_label_and_image_id(self) -> None:
        dataset = CifarImages(self.images, self.labels, [4, 7], randaugment=False)
        image, label, image_id = dataset[(1, 0)]
        self.assertEqual(image.dtype, torch.uint8)
        self.assertTrue(torch.equal(image, torch.from_numpy(self.images[7]).permute(2, 0, 1)))
        self.assertEqual((label, image_id), (7, 7))

    def test_randaugment_depends_only_on_visit_seed(self) -> None:
        dataset = CifarImages(self.images, self.labels, [4], randaugment=True)
        first = [dataset[(0, seed)][0] for seed in range(5)]
        again = [dataset[(0, seed)][0] for seed in range(5)]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, again)))
        self.assertTrue(any(not torch.equal(first[0], other) for other in first[1:]))


class StepBatchSamplerTest(unittest.TestCase):
    def test_each_epoch_visits_every_position_once(self) -> None:
        items = stream(StepBatchSampler(10, 5, seed=0, start_step=0, end_step=4))
        positions = [position for position, _ in items]
        self.assertEqual(sorted(positions[:10]), list(range(10)))
        self.assertEqual(sorted(positions[10:]), list(range(10)))

    def test_resuming_continues_the_same_stream(self) -> None:
        full = stream(StepBatchSampler(10, 4, seed=3, start_step=0, end_step=6))
        resumed = stream(StepBatchSampler(10, 4, seed=3, start_step=2, end_step=6))
        self.assertEqual(resumed, full[8:])

    def test_seed_changes_the_order(self) -> None:
        first = stream(StepBatchSampler(10, 5, seed=0, start_step=0, end_step=2))
        second = stream(StepBatchSampler(10, 5, seed=1, start_step=0, end_step=2))
        self.assertNotEqual(first, second)


class ToModelInputTest(unittest.TestCase):
    def test_resizes_and_normalizes(self) -> None:
        white = torch.full((2, 3, 32, 32), 255, dtype=torch.uint8)
        result = to_model_input(white, 64)
        self.assertEqual(tuple(result.shape), (2, 3, 64, 64))
        expected = [(1 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)]
        for channel, value in enumerate(expected):
            self.assertTrue(torch.allclose(result[:, channel], torch.tensor(value), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
