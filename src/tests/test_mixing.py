"""Tests for the batch-level Mixup and CutMix conditions."""

import unittest

import torch

from src.component.mixing import MIXING, mix_batch


def batch(size: int = 8, num_classes: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 images (B, 3, 32, 32) where image i is filled with the value 10 * i, and labels i mod num_classes."""
    images = torch.stack([torch.full((3, 32, 32), 10 * i, dtype=torch.uint8) for i in range(size)])
    return images, torch.arange(size) % num_classes


class SettingsTest(unittest.TestCase):
    def test_torchvision_reference_values(self) -> None:
        self.assertEqual(MIXING, {"mixup": {"alpha": 0.2}, "cutmix": {"alpha": 1.0}})


class MixupTest(unittest.TestCase):
    def test_images_and_targets_use_the_same_mixing_weight(self) -> None:
        images, labels = batch(num_classes=8)  # label i for image i, filled with 10 * i
        mixed, targets = mix_batch(images, labels, "mixup", seed=0, step=3, num_classes=8)
        self.assertEqual(mixed.shape, (8, 3, 32, 32))
        self.assertEqual(targets.shape, (8, 8))
        self.assertTrue(torch.allclose(targets.sum(1), torch.ones(8)))
        for i in range(8):
            weights = targets[i]
            partner = int(torch.argmax(torch.where(torch.arange(8) == i, -1.0, weights)))
            expected = float(weights[i]) * 10 * i + float(weights[partner]) * 10 * partner
            self.assertAlmostEqual(float(mixed[i, 0, 0, 0]), expected, places=3)

    def test_every_pixel_is_a_blend_of_two_whole_images(self) -> None:
        images, labels = batch()
        mixed, _ = mix_batch(images, labels, "mixup", seed=1, step=0, num_classes=4)
        for i in range(8):
            self.assertEqual(len(torch.unique(mixed[i])), 1)


class CutmixTest(unittest.TestCase):
    def test_a_pasted_box_and_target_weights_match_its_area(self) -> None:
        images, labels = batch(num_classes=8)
        mixed, targets = mix_batch(images, labels, "cutmix", seed=2, step=5, num_classes=8)
        for i in range(8):
            values = torch.unique(mixed[i, 0])
            own = float((mixed[i, 0] == 10 * i).float().mean())
            if len(values) == 2:
                self.assertAlmostEqual(float(targets[i, i]), own, places=5)
            self.assertLessEqual(len(values), 2)
        self.assertTrue(torch.allclose(targets.sum(1), torch.ones(8)))


class DeterminismTest(unittest.TestCase):
    def test_same_seed_and_step_give_the_same_mix(self) -> None:
        images, labels = batch()
        for method in MIXING:
            first = mix_batch(images, labels, method, seed=4, step=7, num_classes=4)
            second = mix_batch(images, labels, method, seed=4, step=7, num_classes=4)
            self.assertTrue(torch.equal(first[0], second[0]) and torch.equal(first[1], second[1]))

    def test_the_random_stream_is_pinned(self) -> None:
        # Changing these draws would make earlier runs impossible to reproduce or resume identically.
        images, labels = batch(num_classes=8)
        expected = {"mixup": (0.867802, [4, 7, 2, 0, 6, 3, 5, 1]), "cutmix": (0.808594, [7, 0, 4, 3, 2, 1, 5, 6])}
        for method, (weight, partners) in expected.items():
            _, targets = mix_batch(images, labels, method, seed=0, step=0, num_classes=8)
            self.assertAlmostEqual(float(targets[0, 0]), weight, places=6)
            found = [i if targets[i, i] == 1 else int(torch.argmax(torch.where(torch.arange(8) == i, -1.0, targets[i])))
                     for i in range(8)]
            self.assertEqual(found, partners, method)

    def test_different_steps_give_different_mixes(self) -> None:
        images, labels = batch()
        a, _ = mix_batch(images, labels, "mixup", seed=4, step=7, num_classes=4)
        b, _ = mix_batch(images, labels, "mixup", seed=4, step=8, num_classes=4)
        self.assertFalse(torch.equal(a, b))

    def test_does_not_touch_the_global_torch_random_state(self) -> None:
        images, labels = batch()
        torch.manual_seed(0)
        expected = torch.rand(1)
        torch.manual_seed(0)
        mix_batch(images, labels, "cutmix", seed=0, step=0, num_classes=4)
        self.assertTrue(torch.equal(torch.rand(1), expected))


class UnknownMethodTest(unittest.TestCase):
    def test_rejects_an_unknown_method(self) -> None:
        images, labels = batch()
        with self.assertRaises(ValueError):
            mix_batch(images, labels, "cutout", seed=0, step=0, num_classes=4)


if __name__ == "__main__":
    unittest.main()
