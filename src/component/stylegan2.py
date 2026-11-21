"""Class-conditional StyleGAN2 networks for 32x32 images (Karras et al., 2020).

Written from the StyleGAN2 and ADA papers, with the settings of ADA's CIFAR
configuration: 512 channels at every resolution, a 2-layer mapping network,
a generator with skip connections to RGB, a discriminator without residual
connections, minibatch standard deviation over groups of 32, and the class
given to the generator as an embedding concatenated with the latent and to the
discriminator by projection.

- Every layer uses equalized learning rate: weights are drawn from N(0, 1)
  and scaled by 1/sqrt(fan-in) when used (and the mapping network's learning
  rate is lowered 100 times).
- Generator: the class embedding and the latent ``z`` are each normalized to
  unit second moment, concatenated and mapped to ``w``. A learned 4x4
  constant is refined at 4, 8, 16 and 32 pixels by 3x3 convolutions whose
  weights are modulated by ``w`` (one style per layer, through a learned
  affine map) and demodulated, plus per-pixel noise with a learned strength.
  Each resolution adds its own RGB output to the upsampled sum of the lower
  ones.
- Discriminator: RGB to features at 32x32, two 3x3 convolutions per
  resolution with a [1, 3, 3, 1] blur before halving, minibatch standard
  deviation, a 3x3 convolution and two dense layers at 4x4. The score is the
  dot product of the last layer with an embedding of the class from an
  8-layer mapping network, divided by sqrt(dimension).

Upsampling is bilinear before the convolution and downsampling a [1, 3, 3, 1]
blur followed by a stride-2 convolution, which for these linear filters equals
the reference implementation's resampling up to the borders. Weights are
32-bit; ``train_gan`` runs the forward passes in bfloat16 on the GPU, as the
reference CIFAR configuration runs every layer in 16-bit floating point.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT2 = math.sqrt(2)


def _lrelu(x: torch.Tensor) -> torch.Tensor:
    """Leaky ReLU (slope 0.2) scaled by sqrt(2) to keep the signal's magnitude."""
    return F.leaky_relu(x, 0.2) * SQRT2


def normalize_2nd_moment(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale each row to unit mean square."""
    return x * (x.square().mean(dim=1, keepdim=True) + eps).rsqrt()


class FullyConnected(nn.Module):
    """Dense layer with equalized learning rate and an optional sqrt(2)-scaled leaky ReLU."""

    def __init__(self, in_features: int, out_features: int, bias_init: float = 0.0, lr_mul: float = 1.0,
                 activation: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.full([out_features], float(bias_init)))
        self.weight_gain = lr_mul / math.sqrt(in_features)
        self.bias_gain = lr_mul
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self.weight * self.weight_gain, self.bias * self.bias_gain)
        return _lrelu(x) if self.activation else x


def _upsample_axis(x: torch.Tensor, dim: int) -> torch.Tensor:
    size = x.shape[dim]
    before = torch.cat([x.narrow(dim, 0, 1), x.narrow(dim, 0, size - 1)], dim)
    after = torch.cat([x.narrow(dim, 1, size - 1), x.narrow(dim, size - 1, 1)], dim)
    return torch.stack([0.75 * x + 0.25 * before, 0.75 * x + 0.25 * after], dim + 1).flatten(dim, dim + 1)


def upsample2x(x: torch.Tensor) -> torch.Tensor:
    """2x bilinear upsampling (``interpolate(align_corners=False)``), with a faster backward pass."""
    return _upsample_axis(_upsample_axis(x, 2), 3)


def _blur_kernel(channels: int, device: torch.device) -> torch.Tensor:
    taps = torch.tensor([1.0, 3.0, 3.0, 1.0], device=device)
    kernel = torch.outer(taps, taps)
    return (kernel / kernel.sum()).expand(channels, 1, 4, 4)


class Conv2dLayer(nn.Module):
    """Discriminator convolution with equalized learning rate, leaky ReLU and optional blur-and-halve."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, down: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.weight_gain = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.down = down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride = 1
        if self.down:  # blur at full resolution, then convolve with stride 2
            x = F.conv2d(F.pad(x, [1, 2, 1, 2]), _blur_kernel(x.shape[1], x.device).to(x.dtype), groups=x.shape[1])
            stride = 2
        x = F.conv2d(x, self.weight * self.weight_gain, stride=stride, padding=self.weight.shape[-1] // 2)
        return _lrelu(x + self.bias.view(1, -1, 1, 1))


def modulated_conv2d(x: torch.Tensor, weight: torch.Tensor, styles: torch.Tensor,
                     demodulate: bool = True) -> torch.Tensor:
    """Convolve each image with the weight scaled by its own style (one scale per input channel).

    With ``demodulate`` each output channel's weights are rescaled to unit
    norm per image, so the output has about unit variance for unit-variance
    input whatever the style.

    Args:
        x: (N, C_in, H, W).
        weight: (C_out, C_in, k, k).
        styles: (N, C_in).
    """
    # Computed without per-image weights: scaling the input channels by the style before a shared
    # convolution and the output channels by the demodulation factor after it is the same operation.
    out = F.conv2d(x * styles[:, :, None, None], weight, padding=weight.shape[-1] // 2)
    if demodulate:
        # sum over (c_in, k, k) of (weight * style)^2, as a product of the squared styles and weights
        norms = styles.square() @ weight.square().sum(dim=(2, 3)).t()
        out = out * (norms + 1e-8).rsqrt()[:, :, None, None]
    return out


class MappingNetwork(nn.Module):
    """Map a latent (optional) and a class label to a style vector."""

    def __init__(self, z_dim: int, num_classes: int, w_dim: int, num_layers: int, lr_mul: float = 0.01) -> None:
        super().__init__()
        self.z_dim, self.num_classes = z_dim, num_classes
        self.embed = FullyConnected(num_classes, w_dim)
        sizes = [z_dim + w_dim] + [w_dim] * num_layers
        self.layers = nn.ModuleList(
            FullyConnected(a, b, lr_mul=lr_mul, activation=True) for a, b in zip(sizes[:-1], sizes[1:])
        )

    def forward(self, z: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
        x = normalize_2nd_moment(self.embed(F.one_hot(labels, self.num_classes).float()))
        if self.z_dim:
            x = torch.cat([normalize_2nd_moment(z), x], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class SynthesisLayer(nn.Module):
    """Modulated 3x3 convolution, noise, bias and leaky ReLU, optionally after 2x upsampling."""

    def __init__(self, in_channels: int, out_channels: int, w_dim: int, resolution: int, up: bool = False) -> None:
        super().__init__()
        self.affine = FullyConnected(w_dim, in_channels, bias_init=1.0)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.noise_strength = nn.Parameter(torch.zeros([]))
        self.resolution, self.up = resolution, up

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.up:
            x = upsample2x(x)
        x = modulated_conv2d(x, self.weight, self.affine(w))
        x = x + noise * self.noise_strength
        return _lrelu(x + self.bias.view(1, -1, 1, 1))


class ToRGB(nn.Module):
    """Modulated 1x1 convolution to RGB, without demodulation or activation."""

    def __init__(self, in_channels: int, w_dim: int) -> None:
        super().__init__()
        self.affine = FullyConnected(w_dim, in_channels, bias_init=1.0)
        self.weight = nn.Parameter(torch.randn(3, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(3))
        self.weight_gain = 1 / math.sqrt(in_channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = modulated_conv2d(x, self.weight, self.affine(w) * self.weight_gain, demodulate=False)
        return x + self.bias.view(1, -1, 1, 1)


class Generator(nn.Module):
    """Class-conditional StyleGAN2 generator; images in about [-1, 1]."""

    def __init__(self, z_dim: int = 512, w_dim: int = 512, num_classes: int = 100, resolution: int = 32,
                 channels: int = 512, mapping_layers: int = 2) -> None:
        super().__init__()
        self.z_dim, self.num_classes, self.resolution = z_dim, num_classes, resolution
        self.mapping = MappingNetwork(z_dim, num_classes, w_dim, mapping_layers)
        self.const = nn.Parameter(torch.randn(channels, 4, 4))
        self.layers = nn.ModuleList([SynthesisLayer(channels, channels, w_dim, 4)])
        self.to_rgb = nn.ModuleList([ToRGB(channels, w_dim)])
        res = 8
        while res <= resolution:
            self.layers.append(SynthesisLayer(channels, channels, w_dim, res, up=True))
            self.layers.append(SynthesisLayer(channels, channels, w_dim, res))
            self.to_rgb.append(ToRGB(channels, w_dim))
            res *= 2

    def make_noise(self, n: int, generator: torch.Generator | None = None,
                   device: torch.device | str = "cpu") -> list[torch.Tensor]:
        """One noise image per synthesis layer, from ``generator`` (on the CPU) or the global generator."""
        return [torch.randn(n, 1, layer.resolution, layer.resolution, generator=generator).to(device)
                if generator is not None else
                torch.randn(n, 1, layer.resolution, layer.resolution, device=device) for layer in self.layers]

    def forward(self, z: torch.Tensor, labels: torch.Tensor, noise: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Images (N, 3, R, R) for latents (N, z_dim) and integer labels (N,); random noise if none is given."""
        if noise is None:
            noise = self.make_noise(len(z), device=z.device)
        w = self.mapping(z, labels)
        x = self.const.unsqueeze(0).expand(len(z), -1, -1, -1)
        x = self.layers[0](x, w, noise[0])
        image = self.to_rgb[0](x, w)
        for block, to_rgb in enumerate(self.to_rgb[1:]):
            first = 1 + 2 * block
            x = self.layers[first](x, w, noise[first])
            x = self.layers[first + 1](x, w, noise[first + 1])
            image = upsample2x(image) + to_rgb(x, w)
        return image


def minibatch_std(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Append the mean standard deviation over groups of images as one extra feature map.

    The group size is the largest divisor of the batch size not above ``group_size``.
    """
    n, c, h, w = x.shape
    group = max(d for d in range(1, min(group_size, n) + 1) if n % d == 0)
    y = x.reshape(group, -1, c, h, w)
    y = (y - y.mean(dim=0)).square().mean(dim=0)
    y = (y + 1e-8).sqrt().mean(dim=(1, 2, 3))
    y = y.reshape(-1, 1, 1, 1).repeat(group, 1, h, w)
    return torch.cat([x, y], dim=1)


class Discriminator(nn.Module):
    """Class-conditional StyleGAN2 discriminator ("orig" architecture) with projection of the class."""

    def __init__(self, num_classes: int = 100, resolution: int = 32, channels: int = 512, mbstd_group: int = 32,
                 mapping_layers: int = 8) -> None:
        super().__init__()
        self.mbstd_group = mbstd_group
        self.from_rgb = Conv2dLayer(3, channels, 1)
        blocks = []
        res = resolution
        while res > 4:
            blocks += [Conv2dLayer(channels, channels, 3), Conv2dLayer(channels, channels, 3, down=True)]
            res //= 2
        self.blocks = nn.Sequential(*blocks)
        self.conv = Conv2dLayer(channels + 1, channels, 3)
        self.fc = FullyConnected(channels * 16, channels, activation=True)
        self.out = FullyConnected(channels, channels)
        self.mapping = MappingNetwork(0, num_classes, channels, mapping_layers)
        self.cmap_dim = channels

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """One score (logit) per image, (N, 1)."""
        x = self.blocks(self.from_rgb(images))
        x = self.conv(minibatch_std(x, self.mbstd_group))
        x = self.out(self.fc(x.flatten(1)))
        cmap = self.mapping(None, labels)
        return (x * cmap).sum(dim=1, keepdim=True) / math.sqrt(self.cmap_dim)
