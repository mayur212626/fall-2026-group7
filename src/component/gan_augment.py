"""Adaptive discriminator augmentation (ADA) for the StyleGAN2 generator.

With 50 real images per class the discriminator memorizes the training set
and GAN training collapses. ADA (Karras et al., 2020) augments every image
the discriminator sees, real and generated, with a probability ``p`` that is
tuned during training: the sign of the discriminator's output on real images,
averaged over a few minibatches (``r_t``), measures overfitting, and ``p``
goes up when ``r_t`` is above a target (0.6) and down when it is below.
Because the generator is scored only through augmented images, augmentations
applied with ``p`` below about 0.8 do not leak into the generated images.

The pipeline is the paper's "bgc" set, applied in this order with the
paper's strengths, each transformation independently per image with
probability ``p``:

- pixel blitting: x-flip, rotation by multiples of 90 degrees, integer
  translation (up to 1/8 of the image);
- geometric: isotropic scaling (lognormal, std 0.2 octaves), rotation
  (uniform, each of two rotations with probability ``1 - sqrt(1 - p)``),
  anisotropic scaling (std 0.2 octaves), fractional translation (std 1/8);
- color: brightness (std 0.2), contrast (lognormal, std 0.5 octaves), luma
  flip, hue rotation (uniform), saturation (lognormal, std 1 octave).

The geometric transformations are combined into one matrix per image and
applied by bilinear resampling with reflection at the borders, only to the
images that were transformed. Unlike the reference implementation, images
are resampled at their own resolution rather than at twice the resolution.
The R1 penalty takes the gradient of the discriminator through this pipeline,
which needs a second derivative of the resampling; PyTorch's ``grid_sample``
has none, so ``bilinear_sample`` does the same computation with ``gather``.
"""

import math

import torch
import torch.nn as nn


def _reflect(coord: torch.Tensor, size: int) -> torch.Tensor:
    """Reflect pixel coordinates into the image like ``grid_sample(padding_mode="reflection")``."""
    low, span = -0.5, float(size)
    shifted = (coord - low).abs()
    extra = torch.fmod(shifted, span)
    flips = torch.floor(shifted / span)
    reflected = torch.where(torch.remainder(flips, 2) == 0, extra + low, span - extra + low)
    return reflected.clamp(0, size - 1)


def bilinear_sample(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinear sampling with reflection padding, differentiable twice with respect to ``image``.

    Same result as ``grid_sample(image, grid, mode="bilinear", padding_mode="reflection",
    align_corners=False)``.

    Args:
        image: (N, C, H, W).
        grid: (N, Ho, Wo, 2) sampling positions in [-1, 1], x first.
    """
    n, c, h, w = image.shape
    with torch.no_grad():
        x = _reflect(((grid[..., 0] + 1) * w - 1) / 2, w)
        y = _reflect(((grid[..., 1] + 1) * h - 1) / 2, h)
        x0, y0 = x.floor(), y.floor()
        wx, wy = (x - x0).unsqueeze(1), (y - y0).unsqueeze(1)
        x0, y0 = x0.long(), y0.long()
        x1, y1 = (x0 + 1).clamp(max=w - 1), (y0 + 1).clamp(max=h - 1)
    flat = image.reshape(n, c, h * w)

    def corner(yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
        index = (yy * w + xx).reshape(n, 1, -1).expand(n, c, -1)
        return torch.gather(flat, 2, index).reshape(n, c, *grid.shape[1:3])

    top = corner(y0, x0) * (1 - wx) + corner(y0, x1) * wx
    bottom = corner(y1, x0) * (1 - wx) + corner(y1, x1) * wx
    return top * (1 - wy) + bottom * wy


def _matrix(rows: list[list], n: int, device: torch.device) -> torch.Tensor:
    """Stack per-image matrix entries (tensors of shape (n,) or numbers) into (n, R, C)."""
    entries = [[e if torch.is_tensor(e) else torch.full((n,), float(e), device=device) for e in row] for row in rows]
    return torch.stack([torch.stack(row, dim=-1) for row in entries], dim=-2)


def _translate2d(tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    return _matrix([[1, 0, tx], [0, 1, ty], [0, 0, 1]], len(tx), tx.device)


def _scale2d(sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
    return _matrix([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], len(sx), sx.device)


def _rotate2d(theta: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(theta), torch.sin(theta)
    return _matrix([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], len(theta), theta.device)


def luma_axis(device: torch.device | str = "cpu") -> torch.Tensor:
    """The gray axis of RGB space as a homogeneous 4-vector of unit length."""
    return torch.tensor([1.0, 1.0, 1.0, 0.0], device=device) / math.sqrt(3)


def rotate3d(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """(N, 4, 4) homogeneous rotations by ``theta`` about the unit ``axis`` (Rodrigues' formula)."""
    vx, vy, vz = (float(a) for a in axis[:3])
    cos, sin = torch.cos(theta), torch.sin(theta)
    one = 1 - cos
    return _matrix([
        [vx * vx * one + cos, vx * vy * one - vz * sin, vx * vz * one + vy * sin, 0],
        [vy * vx * one + vz * sin, vy * vy * one + cos, vy * vz * one - vx * sin, 0],
        [vz * vx * one - vy * sin, vz * vy * one + vx * sin, vz * vz * one + cos, 0],
        [0, 0, 0, 1],
    ], len(theta), theta.device)


def adjust_p(p: float, rt: float, target: float, images: int, speed_kimg: float) -> float:
    """New augmentation probability after ``images`` images with overfitting measure ``rt``.

    ``p`` moves by ``images / (speed_kimg * 1000)`` towards more augmentation
    when ``rt`` is above ``target`` and towards less when it is below, so it
    can go from 0 to 1 in ``speed_kimg`` thousand images. It never drops below 0.
    """
    step = math.copysign(images / (speed_kimg * 1000), rt - target) if rt != target else 0.0
    return max(p + step, 0.0)


class AugmentPipe(nn.Module):
    """The "bgc" augmentations, each applied with probability ``p`` (a buffer, saved with the state)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("p", torch.zeros([]))
        self.xint_max, self.scale_std, self.rotate_max = 0.125, 0.2, 1.0
        self.aniso_std, self.xfrac_std = 0.2, 0.125
        self.brightness_std, self.contrast_std, self.hue_max, self.saturation_std = 0.2, 0.5, 1.0, 1.0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Augment (N, 3, H, W) images with values in about [-1, 1]."""
        p = float(self.p)
        if p <= 0:
            return images
        n, _, h, w = images.shape
        device = images.device

        def chosen(prob: float = p) -> torch.Tensor:
            return torch.rand(n, device=device) < prob

        def keep(values: torch.Tensor, mask: torch.Tensor, identity: float) -> torch.Tensor:
            return torch.where(mask, values, torch.full_like(values, identity))

        # Geometric transformations, as the inverse matrix that maps output to input coordinates.
        g_inv = torch.eye(3, device=device).expand(n, 3, 3)
        flip = keep(torch.floor(torch.rand(n, device=device) * 2), chosen(), 0)
        g_inv = g_inv @ _scale2d(1 - 2 * flip, torch.ones(n, device=device))
        quarter = keep(torch.floor(torch.rand(n, device=device) * 4), chosen(), 0)
        g_inv = g_inv @ _rotate2d(math.pi / 2 * quarter)
        mask = chosen()
        tx = keep(torch.round((torch.rand(n, device=device) * 2 - 1) * self.xint_max * w), mask, 0) * 2 / w
        ty = keep(torch.round((torch.rand(n, device=device) * 2 - 1) * self.xint_max * h), mask, 0) * 2 / h
        g_inv = g_inv @ _translate2d(-tx, -ty)
        scale = keep(torch.exp2(torch.randn(n, device=device) * self.scale_std), chosen(), 1)
        g_inv = g_inv @ _scale2d(1 / scale, 1 / scale)
        p_rot = 1 - math.sqrt(min(max(1 - p, 0.0), 1.0))
        theta = keep((torch.rand(n, device=device) * 2 - 1) * math.pi * self.rotate_max, chosen(p_rot), 0)
        g_inv = g_inv @ _rotate2d(theta)
        aniso = keep(torch.exp2(torch.randn(n, device=device) * self.aniso_std), chosen(), 1)
        g_inv = g_inv @ _scale2d(1 / aniso, aniso)
        theta = keep((torch.rand(n, device=device) * 2 - 1) * math.pi * self.rotate_max, chosen(p_rot), 0)
        g_inv = g_inv @ _rotate2d(theta)
        mask = chosen()
        fx = keep(torch.randn(n, device=device) * self.xfrac_std, mask, 0) * 2
        fy = keep(torch.randn(n, device=device) * self.xfrac_std, mask, 0) * 2
        g_inv = g_inv @ _translate2d(-fx, -fy)

        moved = (g_inv - torch.eye(3, device=device)).abs().amax(dim=(1, 2)) > 1e-6
        if moved.any():
            grid = torch.nn.functional.affine_grid(g_inv[:, :2, :], list(images.shape), align_corners=False)
            sampled = bilinear_sample(images, grid)
            images = torch.where(moved.view(n, 1, 1, 1), sampled, images)

        # Color transformations, as one homogeneous 4x4 matrix per image.
        axis = luma_axis(device)
        outer = torch.outer(axis, axis)
        eye4 = torch.eye(4, device=device)
        color = eye4.expand(n, 4, 4)
        bright = keep(torch.randn(n, device=device) * self.brightness_std, chosen(), 0)
        color = _matrix([[1, 0, 0, bright], [0, 1, 0, bright], [0, 0, 1, bright], [0, 0, 0, 1]], n, device) @ color
        contrast = keep(torch.exp2(torch.randn(n, device=device) * self.contrast_std), chosen(), 1)
        color = _matrix([[contrast, 0, 0, 0], [0, contrast, 0, 0], [0, 0, contrast, 0], [0, 0, 0, 1]], n,
                        device) @ color
        luma_flip = keep(torch.floor(torch.rand(n, device=device) * 2), chosen(), 0)
        color = (eye4 - 2 * outer * luma_flip.view(n, 1, 1)) @ color
        hue = keep((torch.rand(n, device=device) * 2 - 1) * math.pi * self.hue_max, chosen(), 0)
        color = rotate3d(axis, hue) @ color
        saturation = keep(torch.exp2(torch.randn(n, device=device) * self.saturation_std), chosen(), 1)
        color = (outer + (eye4 - outer) * saturation.view(n, 1, 1)) @ color
        pixels = images.reshape(n, 3, h * w)
        pixels = color[:, :3, :3] @ pixels + color[:, :3, 3:]
        return pixels.reshape(n, 3, h, w)
