"""Generated image pool from a trained class-conditional StyleGAN2-ADA generator (condition ``gan``).

The generator is the exponential moving average of the best snapshot of a
``train_gan`` run (lowest FID). Image ``i`` of class ``c`` comes from its own
latent and noise images, drawn from a PyTorch generator seeded with
``image_seed(generation seed, c, i)`` (the rule of the Stable Diffusion
pools), so every image can be regenerated on its own and the first ``k``
images of a class are the same for every pool size. Images are sampled
without truncation (psi = 1), so the pool keeps the generator's full
diversity, and are already 32x32.

Outputs in ``--output-dir``:

    pool.npz    the pool (see ``synthetic_pool``)
    pool.json   generator run, best snapshot, sampling settings, versions and the pool SHA-256

and the generator's row in ``src/component/configs/generator_registry.csv``.

    python -m src.component.generate_gan --run-dir runs/gan/stylegan2ada_b50_s0 \
        --output-dir data/synthetic/stylegan2ada_b50
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.component.generate_sd import image_seed
from src.component.generator_registry import REGISTRY, set_row
from src.component.stylegan2 import Generator
from src.component.synthetic_pool import save_pool

GENERATOR_ID = "gan_stylegan2ada_b50"


def to_uint8(images: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) images in [-1, 1] to (N, H, W, 3) uint8, mapping -1 to 0 and 1 to 255."""
    return (images * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def fixed_inputs(g: Generator, labels: np.ndarray, indices: np.ndarray,
                 seed: int) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Latents, labels and noise images (on the CPU) for images ``indices`` of classes ``labels``."""
    latents, noises = [], []
    for label, index in zip(labels, indices, strict=True):
        rng = torch.Generator().manual_seed(image_seed(seed, int(label), int(index)))
        latents.append(torch.randn(1, g.z_dim, generator=rng))
        noises.append(g.make_noise(1, rng))
    noise = [torch.cat([n[layer] for n in noises]) for layer in range(len(g.layers))]
    return torch.cat(latents), torch.as_tensor(np.asarray(labels), dtype=torch.long), noise


@torch.no_grad()
def render(g: Generator, z: torch.Tensor, labels: torch.Tensor, noise: list[torch.Tensor], device: torch.device,
           batch_size: int) -> np.ndarray:
    """Run the generator over fixed inputs in batches; (N, H, W, 3) uint8."""
    g.eval()
    out = []
    for start in range(0, len(z), batch_size):
        part = slice(start, start + batch_size)
        images = g(z[part].to(device), labels[part].to(device), noise=[n[part].to(device) for n in noise])
        out.append(to_uint8(images))
    return np.concatenate(out)


def generate_images(g: Generator, labels: np.ndarray, indices: np.ndarray, seed: int, device: torch.device,
                    batch_size: int) -> np.ndarray:
    """Images ``indices`` of classes ``labels``; each depends only on (seed, class, index)."""
    return render(g, *fixed_inputs(g, labels, indices, seed), device, batch_size)


def generate_pool(g: Generator, num_classes: int, per_class: int, seed: int, device: torch.device,
                  batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``per_class`` images of every class, sorted by class and index: ``(images, labels, index_in_class)``."""
    labels = np.repeat(np.arange(num_classes), per_class)
    index = np.tile(np.arange(per_class), num_classes)
    return generate_images(g, labels, index, seed, device, batch_size), labels, index


def load_generator(run_dir: Path, device: torch.device) -> tuple[Generator, dict, dict]:
    """The best snapshot's averaged generator of a finished run, its settings and its ``best.pt`` record."""
    from src.component.train_gan import build_generator

    settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"]
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    g = build_generator(settings)
    g.load_state_dict(best["G_ema"])
    return g.to(device).eval(), settings, best


def main() -> None:
    """Sample the pool of a finished ``train_gan`` run and record it."""
    parser = argparse.ArgumentParser(description="Generate a CIFAR-100 image pool with a trained StyleGAN2-ADA.")
    parser.add_argument("--run-dir", type=Path, required=True, help="finished train_gan run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=450)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--generator-id", default=GENERATOR_ID)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()

    from src.component.train import _git_commit

    result = json.loads((args.run_dir / "result.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, settings, best = load_generator(args.run_dir, device)
    started = time.perf_counter()
    images, labels, index = generate_pool(g, settings["num_classes"], args.per_class, args.generation_seed, device,
                                          args.batch_size)
    seconds = time.perf_counter() - started
    pool_path = args.output_dir / "pool.npz"
    sha = save_pool(pool_path, images, labels, index)
    meta = {
        "generator": args.generator_id,
        "model": "StyleGAN2-ADA, class-conditional, trained from scratch",
        "run": str(args.run_dir),
        "budget": settings["budget"],
        "train_seed": settings["seed"],
        "best_kimg": best["kimg"],
        "best_fid": best["fid"],
        "reload_check_passed": result["reload_check_passed"],
        "per_class": args.per_class,
        "images": len(images),
        "truncation_psi": 1.0,
        "generation_seed": args.generation_seed,
        "seed_rule": "torch.Generator seeded with numpy SeedSequence([generation_seed, label, index]), uint32; "
                     "latent first, then one noise image per synthesis layer",
        "generation_seconds": seconds,
        "pool_sha256": sha,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_commit(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    (args.output_dir / "pool.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    set_row(args.registry, args.generator_id, {
        "condition": "gan",
        "base_model": "StyleGAN2-ADA (class-conditional, CIFAR configuration)",
        "revision": (meta["git"]["commit"] or "")[:12],
        "adaptation": f"trained from scratch, {settings['kimg']} kimg, batch {settings['batch_size']}, "
                      f"ADA target {settings['ada_target']}, R1 gamma {settings['r1_gamma']}; "
                      f"best snapshot {best['kimg']} kimg (FID {best['fid']:.2f} on 5,000 images)",
        "adaptation_data": f"training images of budget {settings['budget']} ({result['train_images']} images)",
        "sampler": "one generator pass at 32x32, no truncation",
        "generation_seed": f"{args.generation_seed} (SeedSequence([seed, label, index]))",
        "images_per_class": args.per_class,
        "generation_minutes": round(seconds / 60, 1),
        "status": f"generated {meta['finished'][:10]}, pool sha256 {sha[:12]}",
    })
    print(f"saved {pool_path} ({len(images)} images, sha256 {sha[:12]}) in {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
