"""Train a class-conditional StyleGAN2-ADA on the real training images of one data budget (condition ``gan``).

The generator sees only the budget's training images (50 per class by
default, 5,000 images), never validation or test images. Settings follow
ADA's CIFAR configuration (Karras et al., 2020): batch 64, Adam with
learning rate 0.0025 and betas (0, 0.99), non-saturating logistic loss, R1
penalty on real images with gamma 0.01 every 16 minibatches (lazy
regularization, with the learning rate and betas rescaled by 16/17 for the
discriminator and 4/5 for the generator), no path-length regularization and
no style mixing, an exponential moving average of the generator weights with
a half-life of 500k images (ramped up over the first 5% of the images seen),
and adaptive discriminator augmentation (``gan_augment``) with target 0.6,
updated every 4 minibatches, able to go from 0 to 1 in 500k images.

Every ``snapshot_kimg`` thousand images the averaged generator draws a fixed
set of ``fid_per_class`` images per class (seeds as in ``generate_gan``) and
its FID against the budget's training images is computed (clean-fid
Inception features, as in ``image_quality``). The snapshot with the lowest
FID is kept as ``best.pt``; at the end it is reloaded and must draw the same
images bit for bit. Real batches visit the images in a seeded random order,
epoch by epoch; latents, fake labels and augmentations come from PyTorch's
generator, whose state is saved in every checkpoint, so an interrupted run
resumes where it stopped. Outputs in the run directory:

    config.json     settings and metadata (data, git commit, versions)
    history.jsonl   one line per snapshot: losses, augmentation p, r_t, FID, time
    train.log       the same as text
    fakes_KIMG.png  10 images of each of the first 10 classes per snapshot
    best.pt         averaged generator of the best snapshot
    result.json     best snapshot, its FID, reload check, cost

    python -m src.component.train_gan --budget 50 --seed 0 --kimg 10000
"""

import argparse
import hashlib
import json
import logging
import math
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.component.gan_augment import AugmentPipe, adjust_p
from src.component.generate_gan import fixed_inputs, render
from src.component.stylegan2 import Discriminator, Generator
from src.component.train import LOG_FORMAT, _atomic_save, _git_commit, _trim_history, _write_json, set_seed

FID_SEED = 1_000_003  # seed of the fixed FID images; pools use other seeds (generate_gan, default 0)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_settings(budget: str, seed: int, kimg: float, snapshot_kimg: float) -> dict:
    """Settings of one run: ADA's CIFAR configuration for 100 classes at 32x32."""
    return {
        "budget": budget, "seed": seed, "kimg": kimg, "snapshot_kimg": snapshot_kimg,
        "num_classes": 100, "resolution": 32, "channels": 512, "z_dim": 512, "w_dim": 512,
        "mapping_layers": 2, "d_mapping_layers": 8, "mbstd_group": 32, "batch_size": 64,
        "lr": 0.0025, "betas": [0.0, 0.99], "r1_gamma": 0.01, "r1_interval": 16, "g_reg_interval": 4,
        "ema_kimg": 500, "ema_rampup": 0.05, "ada_target": 0.6, "ada_interval": 4, "ada_kimg": 500,
        "fid_per_class": 50, "loss": "non-saturating logistic",
        "precision": "bfloat16 autocast for training forward passes on the GPU; float32 weights and FID images",
    }


def batch_positions(num_items: int, batch_size: int, iteration: int, seed: int) -> np.ndarray:
    """Positions of the real images in minibatch ``iteration``.

    Images are visited in epochs; epoch ``e`` is a permutation drawn from
    ``(seed, e)``, so every image is used once per epoch and any minibatch can
    be rebuilt from its number alone.
    """
    ks = np.arange(iteration * batch_size, (iteration + 1) * batch_size)
    epochs = ks // num_items
    orders = {e: np.random.default_rng([seed, int(e)]).permutation(num_items) for e in np.unique(epochs)}
    return np.array([orders[e][k % num_items] for e, k in zip(epochs, ks)])


def ema_beta(batch_size: int, nimg: int, ema_kimg: float, rampup: float | None) -> float:
    """Weight of the old average: half-life of ``ema_kimg`` thousand images, at most ``rampup`` x images seen."""
    ema_nimg = ema_kimg * 1000
    if rampup is not None:
        ema_nimg = min(ema_nimg, nimg * rampup)
    return 0.5 ** (batch_size / max(ema_nimg, 1e-8))


def lazy_regularization(lr: float, betas: tuple[float, float], interval: int) -> tuple[float, tuple[float, float]]:
    """Learning rate and Adam betas when a regularizer runs every ``interval`` minibatches."""
    ratio = interval / (interval + 1)
    return lr * ratio, (betas[0] ** ratio, betas[1] ** ratio)


def build_generator(settings: dict) -> Generator:
    return Generator(settings["z_dim"], settings["w_dim"], settings["num_classes"], settings["resolution"],
                     settings["channels"], settings["mapping_layers"])


@dataclass
class GanState:
    """Everything that changes during training and is saved for resuming."""

    g: Generator
    d: Discriminator
    g_ema: Generator
    augment: AugmentPipe
    opt_g: torch.optim.Adam
    opt_d: torch.optim.Adam
    iteration: int = 0
    nimg: int = 0
    best: dict | None = None
    ada_signs: list = field(default_factory=lambda: [0.0, 0])
    last_rt: float | None = None


def new_state(settings: dict, device: torch.device) -> GanState:
    """Fresh networks and optimizers; the generator average starts as a copy of the generator."""
    g = build_generator(settings).to(device)
    d = Discriminator(settings["num_classes"], settings["resolution"], settings["channels"],
                      settings["mbstd_group"], settings["d_mapping_layers"]).to(device)
    g_ema = build_generator(settings).to(device).eval()
    g_ema.load_state_dict(g.state_dict())
    g_ema.requires_grad_(False)
    lr_g, betas_g = lazy_regularization(settings["lr"], tuple(settings["betas"]), settings["g_reg_interval"])
    lr_d, betas_d = lazy_regularization(settings["lr"], tuple(settings["betas"]), settings["r1_interval"])
    return GanState(g, d, g_ema, AugmentPipe().to(device),
                    torch.optim.Adam(g.parameters(), lr=lr_g, betas=betas_g, eps=1e-8),
                    torch.optim.Adam(d.parameters(), lr=lr_d, betas=betas_d, eps=1e-8))


def save_checkpoint(state: GanState, path: Path, extra: dict) -> None:
    """Save networks, optimizers, augmentation, counters and random-generator states."""
    _atomic_save({
        "G": state.g.state_dict(), "D": state.d.state_dict(), "G_ema": state.g_ema.state_dict(),
        "augment": state.augment.state_dict(), "opt_G": state.opt_g.state_dict(), "opt_D": state.opt_d.state_dict(),
        "iteration": state.iteration, "nimg": state.nimg, "best": state.best, "ada_signs": state.ada_signs,
        "last_rt": state.last_rt, "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "extra": extra,
    }, Path(path))


def load_checkpoint(state: GanState, path: Path) -> dict:
    """Restore ``state`` and the random-generator states from ``path``; return ``extra``."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state.g.load_state_dict(checkpoint["G"])
    state.d.load_state_dict(checkpoint["D"])
    state.g_ema.load_state_dict(checkpoint["G_ema"])
    state.augment.load_state_dict(checkpoint["augment"])
    state.opt_g.load_state_dict(checkpoint["opt_G"])
    state.opt_d.load_state_dict(checkpoint["opt_D"])
    state.iteration, state.nimg, state.best = checkpoint["iteration"], checkpoint["nimg"], checkpoint["best"]
    state.ada_signs, state.last_rt = checkpoint["ada_signs"], checkpoint["last_rt"]
    torch.set_rng_state(checkpoint["rng_cpu"])
    if checkpoint["rng_cuda"]:
        torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    return checkpoint["extra"]


def train_iteration(state: GanState, reals: torch.Tensor, labels: torch.Tensor, settings: dict) -> dict:
    """One minibatch: generator step, discriminator step, R1 step when due, average and ADA update."""
    n, device = len(reals), reals.device
    g, d, augment = state.g, state.d, state.augment

    def amp() -> torch.autocast:
        """bfloat16 forward passes on the GPU (weights and optimizer stay 32-bit); full precision on the CPU."""
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")

    g.requires_grad_(True)
    d.requires_grad_(False)
    z = torch.randn(n, settings["z_dim"], device=device)
    c = torch.randint(0, settings["num_classes"], (n,), device=device)
    with amp():
        loss_g = F.softplus(-d(augment(g(z, c).float()), c).float()).mean()
    state.opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    state.opt_g.step()

    g.requires_grad_(False)
    d.requires_grad_(True)
    z = torch.randn(n, settings["z_dim"], device=device)
    c = torch.randint(0, settings["num_classes"], (n,), device=device)
    with amp():
        with torch.no_grad():
            fakes = g(z, c).float()
        real_logits = d(augment(reals), labels).float()
        loss_d = F.softplus(d(augment(fakes), c).float()).mean() + F.softplus(-real_logits).mean()
    state.opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    state.opt_d.step()

    r1 = None
    if state.iteration % settings["r1_interval"] == 0:
        reals_grad = reals.detach().requires_grad_(True)
        with amp():
            logits = d(augment(reals_grad), labels).float()
        (grad,) = torch.autograd.grad(logits.sum(), reals_grad, create_graph=True)
        r1 = grad.float().square().sum(dim=(1, 2, 3)).mean()
        state.opt_d.zero_grad(set_to_none=True)
        (r1 * settings["r1_gamma"] / 2 * settings["r1_interval"]).backward()
        state.opt_d.step()

    beta = ema_beta(n, state.nimg, settings["ema_kimg"], settings["ema_rampup"])
    with torch.no_grad():
        for p_ema, p in zip(state.g_ema.parameters(), g.parameters()):
            p_ema.copy_(p.lerp(p_ema, beta))
    state.nimg += n
    state.iteration += 1

    state.ada_signs[0] += real_logits.detach().sign().sum().item()
    state.ada_signs[1] += n
    if state.iteration % settings["ada_interval"] == 0:
        rt = state.ada_signs[0] / state.ada_signs[1]
        state.augment.p.fill_(adjust_p(float(state.augment.p), rt, settings["ada_target"],
                                       n * settings["ada_interval"], settings["ada_kimg"]))
        state.ada_signs = [0.0, 0]
        state.last_rt = rt
    return {"loss_g": loss_g.item(), "loss_d": loss_d.item(), "r1": None if r1 is None else r1.item()}


def save_grid(images: np.ndarray, per_class: int, path: Path, classes: int = 10, columns: int = 10) -> None:
    """Save ``columns`` images of each of the first ``classes`` classes as one PNG (a row per class)."""
    classes, columns = min(classes, len(images) // per_class), min(columns, per_class)
    rows = [np.concatenate(list(images[c * per_class:c * per_class + columns]), axis=1) for c in range(classes)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def run_training(settings: dict, metadata: dict, real_images: np.ndarray, real_labels: np.ndarray, run_dir: Path,
                 device: torch.device, fid_fn: Callable[[np.ndarray], float]) -> dict:
    """Train one GAN to completion, resuming from ``run_dir/last.pt`` if present.

    Args:
        real_images: (N, R, R, 3) uint8 training images of the budget.
        real_labels: (N,) their classes.
        fid_fn: FID of (M, R, R, 3) uint8 generated images against the training images.

    Raises:
        FileExistsError: If the run already finished, or the directory holds
            files but no ``config.json``.
        ValueError: If the saved settings differ from ``settings``.
    """
    run_dir = Path(run_dir)
    config_path, last_path, history_path = run_dir / "config.json", run_dir / "last.pt", run_dir / "history.jsonl"
    if (run_dir / "result.json").exists():
        raise FileExistsError(f"{run_dir} is already complete")
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8"))["settings"] != settings:
            raise ValueError(f"settings differ from the saved run in {run_dir}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} has files but no config.json")

    set_seed(settings["seed"])
    state = new_state(settings, device)
    if last_path.exists():
        extra = load_checkpoint(state, last_path)
        _trim_history(history_path, state.iteration)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            _write_json(config_path, {"settings": settings, "metadata": metadata})
        _trim_history(history_path, 0)
        extra = {"elapsed": 0.0, "peak_memory_mb": 0.0}

    reals = torch.from_numpy(real_images).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(device)
    labels = torch.as_tensor(np.asarray(real_labels), dtype=torch.long, device=device)
    per_class = settings["fid_per_class"]
    fid_labels = np.repeat(np.arange(settings["num_classes"]), per_class)
    fid_inputs = fixed_inputs(state.g_ema, fid_labels, np.tile(np.arange(per_class), settings["num_classes"]),
                              FID_SEED)
    batch = settings["batch_size"]
    total = math.ceil(settings["kimg"] * 1000 / batch)
    interval = math.ceil(settings["snapshot_kimg"] * 1000 / batch)

    handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    verb = "resuming from" if state.iteration else "starting at"
    logger.info("%s: %s minibatch %d of %d (%d real images)", run_dir.name, verb, state.iteration, total,
                len(reals))
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        while state.iteration < total:
            started = time.perf_counter()
            until = min((state.iteration // interval + 1) * interval, total)
            losses = []
            while state.iteration < until:
                positions = torch.as_tensor(batch_positions(len(reals), batch, state.iteration, settings["seed"]),
                                            device=device)
                losses.append(train_iteration(state, reals[positions], labels[positions], settings))
            fakes = render(state.g_ema, *fid_inputs, device, 500)
            fid = fid_fn(fakes)
            extra["elapsed"] += time.perf_counter() - started
            if device.type == "cuda":
                extra["peak_memory_mb"] = max(extra["peak_memory_mb"], torch.cuda.max_memory_allocated(device) / 2**20)

            r1 = [x["r1"] for x in losses if x["r1"] is not None]
            record = {
                "step": state.iteration, "iteration": state.iteration, "kimg": state.nimg / 1000,
                "loss_g": float(np.mean([x["loss_g"] for x in losses])),
                "loss_d": float(np.mean([x["loss_d"] for x in losses])),
                "r1": float(np.mean(r1)) if r1 else None,
                "augment_p": float(state.augment.p), "rt": state.last_rt,
                "fid": fid, "elapsed_seconds": extra["elapsed"],
            }
            with history_path.open("a", encoding="utf-8") as history:
                history.write(json.dumps(record) + "\n")
            logger.info("kimg %.1f: G loss %.3f, D loss %.3f, p %.3f, FID %.2f, %.1f min", record["kimg"],
                        record["loss_g"], record["loss_d"], record["augment_p"], fid, extra["elapsed"] / 60)
            save_grid(fakes, per_class, run_dir / f"fakes_{state.nimg // 1000:06d}.png")
            if state.best is None or fid < state.best["fid"]:
                sha = hashlib.sha256(fakes.tobytes()).hexdigest()
                state.best = {"kimg": record["kimg"], "iteration": state.iteration, "fid": fid, "fakes_sha256": sha}
                _atomic_save({"G_ema": state.g_ema.state_dict(), **state.best}, run_dir / "best.pt")
            save_checkpoint(state, last_path, extra)

        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        g = build_generator(settings).to(device)
        g.load_state_dict(best["G_ema"])
        again = hashlib.sha256(render(g, *fid_inputs, device, 500).tobytes()).hexdigest()
        result = {
            "run": run_dir.name,
            "train_images": len(reals),
            "best_kimg": best["kimg"],
            "best_fid": best["fid"],
            "reload_check_passed": again == best["fakes_sha256"],
            "final_augment_p": float(state.augment.p),
            "kimg": state.nimg / 1000,
            "cost": {"train_seconds": extra["elapsed"], "peak_memory_mb": extra["peak_memory_mb"]},
        }
        _write_json(run_dir / "result.json", result)
        last_path.unlink()
        logger.info("%s done: best snapshot %.1f kimg, FID %.2f, %.1f h", run_dir.name, best["kimg"], best["fid"],
                    extra["elapsed"] / 3600)
    finally:
        logger.removeHandler(handler)
        handler.close()
    return result


def main() -> None:
    """Load the budget's training images, set up FID and train one GAN."""
    parser = argparse.ArgumentParser(description="Train a class-conditional StyleGAN2-ADA on one CIFAR-100 budget.")
    parser.add_argument("--budget", choices=["5", "10", "20", "50", "full"], default="50")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kimg", type=float, required=True, help="thousands of real images shown to D")
    parser.add_argument("--snapshot-kimg", type=float, default=200)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/gan"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--manifest", type=Path, default=Path("src/component/configs/splits/cifar100.json"))
    parser.add_argument("--fid-batch", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT)

    from cleanfid.features import build_feature_extractor

    from src.component.cifar_splits import labels_sha256
    from src.component.data import budget_indices, load_cifar100, load_manifest
    from src.component.image_quality import frechet_distance, image_features

    images, labels = load_cifar100(args.data_root, train=True)
    manifest = load_manifest(args.manifest)
    if labels_sha256(labels) != manifest["labels_sha256"]:
        raise ValueError("CIFAR-100 labels do not match the split manifest")
    indices = np.asarray(budget_indices(manifest, args.budget))
    reals, real_labels = images[indices], np.asarray(labels)[indices]

    device = torch.device("cuda")
    settings = make_settings(args.budget, args.seed, args.kimg, args.snapshot_kimg)
    run_dir = args.runs_root / f"stylegan2ada_b{args.budget}_s{args.seed}"
    model = build_feature_extractor("clean", device, use_dataparallel=False)
    real_features = image_features(reals, model, device, args.fid_batch)

    def fid_fn(fakes: np.ndarray) -> float:
        return frechet_distance(real_features, image_features(fakes, model, device, args.fid_batch))

    from importlib.metadata import version

    metadata = {
        "data": {"manifest": str(args.manifest), "train_images": len(indices),
                 "train_indices_sha256": hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest()},
        "fid": "clean-fid clean mode, Inception pool3, generated images vs the budget's training images",
        "git": _git_commit(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "clean_fid": version("clean-fid")},
        "device": torch.cuda.get_device_name(device),
    }
    run_training(settings, metadata, reals, real_labels, run_dir, device, fid_fn)


if __name__ == "__main__":
    main()
