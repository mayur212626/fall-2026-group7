"""Tests for the adaptive discriminator augmentation (CPU, small images)."""

import math
import unittest

import torch

from src.component.gan_augment import AugmentPipe, adjust_p, bilinear_sample, luma_axis, rotate3d


class BilinearSampleTest(unittest.TestCase):
    def test_identity_grid_returns_the_image(self) -> None:
        image = torch.randn(2, 3, 8, 8)
        grid = torch.nn.functional.affine_grid(torch.eye(2, 3).expand(2, 2, 3), (2, 3, 8, 8), align_corners=False)
        torch.testing.assert_close(bilinear_sample(image, grid), image)

    def test_matches_grid_sample_with_reflection(self) -> None:
        torch.manual_seed(0)
        image = torch.randn(2, 3, 8, 8)
        theta = torch.eye(2, 3).expand(2, 2, 3) + 0.3 * torch.randn(2, 2, 3)
        grid = torch.nn.functional.affine_grid(theta, (2, 3, 8, 8), align_corners=False)
        expected = torch.nn.functional.grid_sample(image, grid, mode="bilinear", padding_mode="reflection",
                                                   align_corners=False)
        torch.testing.assert_close(bilinear_sample(image, grid), expected, atol=1e-5, rtol=1e-5)

    def test_supports_the_second_derivative(self) -> None:
        image = torch.randn(1, 3, 8, 8, requires_grad=True)
        grid = torch.nn.functional.affine_grid(torch.tensor([[[0.9, 0.2, 0.1], [-0.2, 0.9, 0.0]]]), (1, 3, 8, 8),
                                               align_corners=False)
        (grad,) = torch.autograd.grad(bilinear_sample(image, grid).square().sum(), image, create_graph=True)
        grad.square().sum().backward()
        self.assertGreater(image.grad.abs().sum().item(), 0)


class ColorTest(unittest.TestCase):
    def test_rotation_about_the_luma_axis_keeps_gray(self) -> None:
        matrix = rotate3d(luma_axis(), torch.tensor([1.0, -2.0]))
        gray = torch.tensor([0.3, 0.3, 0.3, 1.0])
        torch.testing.assert_close(matrix @ gray, gray.expand(2, 4))


class AugmentPipeTest(unittest.TestCase):
    def test_probability_zero_is_the_identity(self) -> None:
        image = torch.randn(4, 3, 32, 32)
        self.assertTrue(torch.equal(AugmentPipe()(image), image))

    def test_probability_one_changes_images_and_keeps_shape(self) -> None:
        torch.manual_seed(0)
        pipe = AugmentPipe()
        pipe.p.fill_(1.0)
        image = torch.rand(4, 3, 32, 32) * 2 - 1
        out = pipe(image)
        self.assertEqual(out.shape, image.shape)
        self.assertFalse(torch.allclose(out, image))
        self.assertTrue(torch.isfinite(out).all())

    def test_gradient_reaches_the_input_twice(self) -> None:
        torch.manual_seed(0)
        pipe = AugmentPipe()
        pipe.p.fill_(0.8)
        image = torch.rand(2, 3, 32, 32, requires_grad=True)
        (grad,) = torch.autograd.grad(pipe(image).square().sum(), image, create_graph=True)
        grad.square().sum().backward()
        self.assertGreater(image.grad.abs().sum().item(), 0)

    def test_p_is_saved_with_the_state_dict(self) -> None:
        pipe = AugmentPipe()
        pipe.p.fill_(0.25)
        other = AugmentPipe()
        other.load_state_dict(pipe.state_dict())
        self.assertEqual(other.p.item(), 0.25)


class AdjustPTest(unittest.TestCase):
    def test_overfitting_raises_p_and_p_never_drops_below_zero(self) -> None:
        step = 64 * 4 / (500 * 1000)
        self.assertAlmostEqual(adjust_p(0.1, rt=0.9, target=0.6, images=64 * 4, speed_kimg=500), 0.1 + step)
        self.assertAlmostEqual(adjust_p(0.1, rt=0.2, target=0.6, images=64 * 4, speed_kimg=500), 0.1 - step)
        self.assertEqual(adjust_p(0.0, rt=0.2, target=0.6, images=64 * 4, speed_kimg=500), 0.0)

    def test_full_speed_moves_p_from_zero_to_one_in_speed_kimg(self) -> None:
        p = 0.0
        for _ in range(math.ceil(500 * 1000 / 256)):
            p = adjust_p(p, rt=1.0, target=0.6, images=256, speed_kimg=500)
        self.assertGreaterEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main()
