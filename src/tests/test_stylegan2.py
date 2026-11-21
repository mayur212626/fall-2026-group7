"""Tests for the conditional StyleGAN2 networks (CPU, few channels)."""

import unittest

import torch

from src.component.stylegan2 import Discriminator, Generator, minibatch_std, modulated_conv2d, upsample2x

SMALL = {"z_dim": 16, "w_dim": 16, "num_classes": 5, "resolution": 32, "channels": 32}


class UpsampleTest(unittest.TestCase):
    def test_equals_bilinear_interpolation(self) -> None:
        x = torch.randn(2, 3, 5, 4)
        expected = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        torch.testing.assert_close(upsample2x(x), expected)


class ModulatedConvTest(unittest.TestCase):
    def test_demodulation_gives_unit_output_variance(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(4, 64, 16, 16)
        weight = torch.randn(64, 64, 3, 3)
        styles = torch.rand(4, 64) * 3 + 0.1
        out = modulated_conv2d(x, weight, styles, demodulate=True)
        self.assertEqual(out.shape, (4, 64, 16, 16))
        self.assertAlmostEqual(out.std().item(), 1.0, delta=0.15)

    def test_each_image_uses_its_own_style(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 8, 8, 8)
        weight = torch.randn(4, 8, 3, 3)
        styles = torch.rand(2, 8) + 0.5
        both = modulated_conv2d(x, weight, styles)
        first = modulated_conv2d(x[:1], weight, styles[:1])
        torch.testing.assert_close(both[:1], first, atol=1e-5, rtol=1e-5)


class GeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.g = Generator(**SMALL).eval()

    def test_output_is_a_batch_of_rgb_images(self) -> None:
        z = torch.randn(3, SMALL["z_dim"])
        out = self.g(z, torch.tensor([0, 1, 4]))
        self.assertEqual(out.shape, (3, 3, 32, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_class_changes_the_image(self) -> None:
        z = torch.randn(1, SMALL["z_dim"]).repeat(2, 1)
        noise = self.g.make_noise(2, torch.Generator().manual_seed(1))
        out = self.g(z, torch.tensor([0, 3]), noise=noise)
        self.assertFalse(torch.allclose(out[0], out[1]))

    def test_given_noise_makes_the_output_fixed(self) -> None:
        with torch.no_grad():
            for layer in self.g.layers:
                layer.noise_strength.fill_(0.5)  # learned during training; starts at 0
        z = torch.randn(2, SMALL["z_dim"])
        labels = torch.tensor([1, 2])
        noise = self.g.make_noise(2, torch.Generator().manual_seed(1))
        torch.testing.assert_close(self.g(z, labels, noise=noise), self.g(z, labels, noise=noise))
        self.assertFalse(torch.allclose(self.g(z, labels), self.g(z, labels)))

    def test_an_image_does_not_depend_on_the_rest_of_the_batch(self) -> None:
        z = torch.randn(4, SMALL["z_dim"])
        labels = torch.tensor([0, 1, 2, 3])
        noise = self.g.make_noise(4, torch.Generator().manual_seed(2))
        batch = self.g(z, labels, noise=noise)
        alone = self.g(z[2:3], labels[2:3], noise=[n[2:3] for n in noise])
        torch.testing.assert_close(batch[2:3], alone, atol=1e-5, rtol=1e-5)


class DiscriminatorTest(unittest.TestCase):
    def test_one_score_per_image_and_the_class_matters(self) -> None:
        torch.manual_seed(0)
        d = Discriminator(num_classes=5, resolution=32, channels=32, mbstd_group=4)
        images = torch.randn(4, 3, 32, 32)
        a = d(images, torch.tensor([0, 1, 2, 3]))
        b = d(images, torch.tensor([4, 4, 4, 4]))
        self.assertEqual(a.shape, (4, 1))
        self.assertFalse(torch.allclose(a, b))

    def test_minibatch_std_uses_a_group_that_divides_the_batch(self) -> None:
        out = minibatch_std(torch.randn(6, 8, 4, 4), group_size=4)
        self.assertEqual(out.shape, (6, 9, 4, 4))

    def test_gradient_reaches_the_generator(self) -> None:
        torch.manual_seed(0)
        g = Generator(**SMALL)
        d = Discriminator(num_classes=5, resolution=32, channels=32, mbstd_group=4)
        labels = torch.tensor([0, 1, 2, 3])
        d(g(torch.randn(4, SMALL["z_dim"]), labels), labels).sum().backward()
        self.assertTrue(all(p.grad is not None for p in g.parameters() if p.requires_grad))


if __name__ == "__main__":
    unittest.main()
