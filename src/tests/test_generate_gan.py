"""Tests for sampling a generated pool from a trained generator (CPU, tiny generator)."""

import unittest

import numpy as np
import torch

from src.component.generate_gan import generate_images, generate_pool, to_uint8
from src.component.stylegan2 import Generator

TINY = {"z_dim": 8, "w_dim": 8, "num_classes": 3, "resolution": 8, "channels": 8}


def tiny_generator() -> Generator:
    torch.manual_seed(0)
    g = Generator(**TINY).eval()
    with torch.no_grad():
        for layer in g.layers:
            layer.noise_strength.fill_(0.5)
    return g


class ToUint8Test(unittest.TestCase):
    def test_maps_minus_one_to_one_onto_0_to_255(self) -> None:
        images = torch.tensor([-1.0, 0.0, 1.0, 2.0]).view(1, 1, 1, 4).expand(1, 3, 1, 4)
        out = to_uint8(images)
        self.assertEqual(out.shape, (1, 1, 4, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out[0, 0, :, 0].tolist(), [0, 128, 255, 255])


class GenerateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.g = tiny_generator()

    def test_an_image_depends_only_on_seed_class_and_index(self) -> None:
        labels, indices = np.array([0, 1, 2]), np.array([0, 0, 4])
        a = generate_images(self.g, labels, indices, seed=7, device=torch.device("cpu"), batch_size=2)
        b = generate_images(self.g, labels[::-1].copy(), indices[::-1].copy(), seed=7, device=torch.device("cpu"),
                            batch_size=3)
        self.assertLessEqual(np.abs(a.astype(int) - b[::-1].astype(int)).max(), 1)
        c = generate_images(self.g, labels, indices, seed=8, device=torch.device("cpu"), batch_size=2)
        self.assertFalse(np.array_equal(a, c))

    def test_pool_is_sorted_and_smaller_pools_are_nested(self) -> None:
        images, labels, index = generate_pool(self.g, num_classes=3, per_class=4, seed=0,
                                              device=torch.device("cpu"), batch_size=5)
        self.assertEqual(images.shape, (12, 8, 8, 3))
        self.assertEqual(labels.tolist(), [0] * 4 + [1] * 4 + [2] * 4)
        self.assertEqual(index.tolist(), [0, 1, 2, 3] * 3)
        small, _, _ = generate_pool(self.g, num_classes=3, per_class=2, seed=0, device=torch.device("cpu"),
                                    batch_size=5)
        self.assertLessEqual(np.abs(small.astype(int) - images[index < 2].astype(int)).max(), 1)


if __name__ == "__main__":
    unittest.main()
