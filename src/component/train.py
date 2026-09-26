"""Training runner for the CIFAR-100 classification benchmark.

One call trains one model, initialization, condition, data budget and seed
for a fixed number of optimizer steps, validating at equal intervals. Each
run writes to its own directory:

    config.json    settings and environment metadata
    history.jsonl  one line per validation
    best.pt        weights of the best validation checkpoint
    last.pt        full training state, removed when the run completes
    result.json    best validation scores and predictions, and measured cost

Running the same command again resumes an interrupted run from ``last.pt``.
The test set is never used here.

Example, from the repository root:

    python -m src.component.train --config src/component/configs/train_pretrained.json \
        --model resnet50 --init pretrained --condition real_only --budget 50 --seed 0
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from src.component.cifar_splits import labels_sha256
from src.component.data import (
    CifarImages,
    StepBatchSampler,
    budget_indices,
    evaluation_batches,
    load_cifar100,
    load_manifest,
    to_model_input,
)
from src.component.metrics import classification_metrics
from src.component.models import build_model, weights_name

SUMMARY_KEYS = ("accuracy", "macro_f1", "balanced_accuracy", "loss")


@dataclass
class TrainState:
    """Everything that changes during training and is saved for resuming."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    step: int = 0
    best: dict | None = None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Learning-rate multiplier for the update at 0-based ``step``.

    Linear warmup to 1 over ``warmup_steps``, then cosine decay towards 0.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def is_better(candidate: dict, best: dict | None) -> bool:
    """Checkpoint rule: higher validation accuracy, then lower validation loss.

    A full tie keeps the earlier checkpoint.
    """
    if best is None:
        return True
    if candidate["accuracy"] != best["accuracy"]:
        return candidate["accuracy"] > best["accuracy"]
    return candidate["loss"] < best["loss"]


def build_optimizer(model: nn.Module, recipe: dict) -> torch.optim.Optimizer:
    """Create the optimizer named in ``recipe`` (``"adamw"`` or ``"sgd"``)."""
    if recipe["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    if recipe["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=recipe["lr"],
            momentum=recipe["momentum"],
            weight_decay=recipe["weight_decay"],
        )
    raise ValueError(f"unknown optimizer {recipe['optimizer']!r}")


def new_state(settings: dict, device: torch.device) -> TrainState:
    """Seed everything and create the model, optimizer and schedule for a run."""
    set_seed(settings["seed"])
    model = build_model(settings["model"], settings["init"] == "pretrained", settings["num_classes"])
    model = model.to(device)
    optimizer = build_optimizer(model, settings["recipe"])
    total = settings["steps"]
    warmup = max(1, round(settings["warmup_fraction"] * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total, warmup))
    return TrainState(model, optimizer, scheduler)


def train_steps(
    state: TrainState,
    dataset: CifarImages,
    settings: dict,
    until_step: int,
    device: torch.device,
) -> float:
    """Train from ``state.step`` up to ``until_step`` and return the mean loss."""
    sampler = StepBatchSampler(len(dataset), settings["batch_size"], settings["seed"], state.step, until_step)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=settings["workers"],
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=settings["label_smoothing"])
    state.model.train()
    total_loss = torch.zeros((), device=device)
    steps = 0
    for images, labels, _ in loader:
        inputs = to_model_input(images.to(device, non_blocking=True), settings["image_size"])
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = criterion(state.model(inputs), labels)
        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(state.model.parameters(), settings["grad_clip"])
        state.optimizer.step()
        state.scheduler.step()
        state.step += 1
        total_loss += loss.detach().float()
        steps += 1
    return total_loss.item() / max(1, steps)


@torch.no_grad()
def evaluate(
    model: nn.Module, dataset: CifarImages, settings: dict, device: torch.device
) -> tuple[dict, list[dict]]:
    """Score ``model`` on ``dataset``: metrics plus mean cross-entropy, and predictions."""
    loader = DataLoader(
        dataset,
        batch_sampler=evaluation_batches(len(dataset), settings["batch_size"]),
        num_workers=settings["workers"],
        pin_memory=device.type == "cuda",
    )
    model.eval()
    loss_sum = 0.0
    true_labels: list[int] = []
    predicted: list[int] = []
    image_ids: list[int] = []
    for images, labels, ids in loader:
        inputs = to_model_input(images.to(device, non_blocking=True), settings["image_size"])
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inputs)
        loss_sum += F.cross_entropy(logits.float(), labels, reduction="sum").item()
        true_labels += labels.tolist()
        predicted += logits.argmax(dim=1).tolist()
        image_ids += ids.tolist()
    metrics = classification_metrics(true_labels, predicted, settings["num_classes"])
    metrics["loss"] = loss_sum / len(dataset)
    predictions = [
        {"image_id": i, "true": t, "pred": p} for i, t, p in zip(image_ids, true_labels, predicted)
    ]
    return metrics, predictions


def _atomic_save(obj: dict, path: Path) -> None:
    """Save with ``torch.save`` to a temporary file, then rename over ``path``."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(state: TrainState, path: Path, extra: dict) -> None:
    """Save the full training state and ``extra`` bookkeeping to ``path``."""
    _atomic_save(
        {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict(),
            "step": state.step,
            "best": state.best,
            "extra": extra,
        },
        Path(path),
    )


def load_checkpoint(state: TrainState, path: Path) -> dict:
    """Restore ``state`` from ``path`` and return the saved ``extra`` dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state.model.load_state_dict(checkpoint["model"])
    state.optimizer.load_state_dict(checkpoint["optimizer"])
    state.scheduler.load_state_dict(checkpoint["scheduler"])
    state.step = checkpoint["step"]
    state.best = checkpoint["best"]
    return checkpoint["extra"]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _trim_history(path: Path, last_step: int) -> None:
    """Drop history lines written after the checkpoint being resumed."""
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if json.loads(line)["step"] <= last_step]
        path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def run_training(
    settings: dict,
    metadata: dict,
    train_set: CifarImages,
    val_set: CifarImages,
    run_dir: Path,
    device: torch.device,
) -> dict:
    """Train one run to completion, resuming from ``run_dir/last.pt`` if present.

    Raises:
        FileExistsError: If the run already finished, or the directory holds
            files but no checkpoint to resume.
        ValueError: If resuming with settings that differ from the saved ones.
    """
    run_dir = Path(run_dir)
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.jsonl"
    if (run_dir / "result.json").exists():
        raise FileExistsError(f"{run_dir} is already complete")

    state = new_state(settings, device)
    if last_path.exists():
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"]
        if saved != settings:
            raise ValueError(f"settings differ from the saved run in {run_dir}")
        extra = load_checkpoint(state, last_path)
        _trim_history(history_path, state.step)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"{run_dir} has files but no checkpoint to resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "config.json", {"settings": settings, "metadata": metadata})
        extra = {"elapsed": 0.0, "peak_memory_mb": 0.0}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    interval = math.ceil(settings["steps"] / settings["evaluations"])
    while state.step < settings["steps"]:
        started = time.perf_counter()
        until = min((state.step // interval + 1) * interval, settings["steps"])
        train_loss = train_steps(state, train_set, settings, until, device)
        metrics, _ = evaluate(state.model, val_set, settings, device)
        extra["elapsed"] += time.perf_counter() - started
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 2**20
            extra["peak_memory_mb"] = max(extra["peak_memory_mb"], peak)

        record = {
            "step": state.step,
            "train_loss": train_loss,
            "lr": state.scheduler.get_last_lr()[0],
            **{f"val_{key}": metrics[key] for key in SUMMARY_KEYS},
            "elapsed_seconds": extra["elapsed"],
        }
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(record) + "\n")

        candidate = {"accuracy": metrics["accuracy"], "loss": metrics["loss"]}
        if is_better(candidate, state.best):
            state.best = {"step": state.step, **candidate}
            _atomic_save({"model": state.model.state_dict(), "step": state.step}, run_dir / "best.pt")
        save_checkpoint(state, last_path, extra)

    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    state.model.load_state_dict(best["model"])
    metrics, predictions = evaluate(state.model, val_set, settings, device)
    result = {
        "run": run_dir.name,
        "best_step": best["step"],
        "best_validation": metrics,
        "reload_check_passed": metrics["accuracy"] == state.best["accuracy"],
        "validation_predictions": predictions,
        "cost": {"train_seconds": extra["elapsed"], "peak_memory_mb": extra["peak_memory_mb"]},
    }
    _write_json(run_dir / "result.json", result)
    last_path.unlink()
    return result


def make_settings(
    config: dict,
    model: str,
    init: str,
    condition: str,
    budget: str,
    seed: int,
    workers: int,
    steps: int | None = None,
    lr: float | None = None,
) -> dict:
    """Combine a training config with one run's choices into run settings.

    ``steps`` and ``lr`` override the config (used by pilot runs); the
    values actually used are stored in the settings and saved with the run.

    Raises:
        ValueError: If the config has no step budget for ``budget`` and
            ``steps`` is not given.
    """
    steps = steps or config["steps"].get(budget)
    if not steps:
        raise ValueError(f"no step budget for budget {budget}; set it in the config or pass steps")
    recipe = dict(config["models"][model])
    if lr is not None:
        recipe["lr"] = lr
    return {
        "model": model,
        "init": init,
        "condition": condition,
        "budget": budget,
        "seed": seed,
        "num_classes": 100,
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "steps": steps,
        "evaluations": config["evaluations"],
        "warmup_fraction": config["warmup_fraction"],
        "grad_clip": config["grad_clip"],
        "label_smoothing": config["label_smoothing"],
        "workers": workers,
        "recipe": recipe,
    }


def _git_commit() -> dict:
    """Return the current commit and whether the working tree has changes."""

    def git(*args: str) -> str | None:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def main() -> None:
    """Parse the command line, load the data and run one training run."""
    parser = argparse.ArgumentParser(description="Train one CIFAR-100 benchmark run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet50", "vit_b_16"], required=True)
    parser.add_argument("--init", choices=["pretrained", "scratch"], required=True)
    parser.add_argument("--condition", choices=["real_only", "randaugment"], required=True)
    parser.add_argument("--budget", choices=["5", "10", "20", "50", "full"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, help="override the config's step budget (pilot runs)")
    parser.add_argument("--lr", type=float, help="override the config's learning rate (pilot runs)")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--manifest", type=Path, default=Path("src/component/configs/splits/cifar100.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        settings = make_settings(
            config, args.model, args.init, args.condition, args.budget, args.seed,
            args.workers, steps=args.steps, lr=args.lr,
        )
    except ValueError as error:
        parser.error(str(error))

    images, labels = load_cifar100(args.data_root, train=True)
    manifest = load_manifest(args.manifest)
    if labels_sha256(labels) != manifest["labels_sha256"]:
        raise ValueError("CIFAR-100 labels do not match the split manifest")
    train_set = CifarImages(
        images, labels, budget_indices(manifest, args.budget), randaugment=args.condition == "randaugment"
    )
    val_set = CifarImages(images, labels, manifest["validation_indices"], randaugment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "weights": weights_name(args.model, args.init == "pretrained"),
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "config_file": str(args.config),
        "git": _git_commit(),
    }
    name = f"{args.init}_{args.model}_{args.condition}_b{args.budget}_s{args.seed}"
    result = run_training(settings, metadata, train_set, val_set, args.runs_root / name, device)
    best = result["best_validation"]
    print(
        f"{name}: best step {result['best_step']}, accuracy {best['accuracy']:.2f}, "
        f"macro-F1 {best['macro_f1']:.2f}, {result['cost']['train_seconds'] / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
