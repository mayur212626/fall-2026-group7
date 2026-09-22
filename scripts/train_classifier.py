import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision.models import get_model
from torchvision.transforms import InterpolationMode, v2

from scripts.checkpoint import load_checkpoint, save_checkpoint
from scripts.metrics import classification_metrics
from scripts.prepare_cifar100 import make_splits


def image_transform(config, randaugment=False):
    transforms = [v2.Resize((config["image_size"], config["image_size"]),
                            interpolation=InterpolationMode.BILINEAR, antialias=True)]
    if randaugment:
        transforms.append(v2.RandAugment(**config["randaugment"],
                                        interpolation=InterpolationMode.BILINEAR, fill=128))
    transforms.extend([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(config["normalization_mean"], config["normalization_std"])])
    return v2.Compose(transforms)


@torch.inference_mode()
def evaluate(model, loader, device="cuda"):
    model.eval()
    loss_sum = 0.0
    targets, predictions = [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        if logits.shape != (len(labels), 100) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite logits for 100 classes.")
        batch_loss = F.cross_entropy(logits, labels, reduction="sum")
        if not torch.isfinite(batch_loss):
            raise FloatingPointError("Validation loss is not finite.")
        loss_sum += batch_loss.item()
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    scores = classification_metrics(targets, predictions, 100)
    scores["validation_loss"] = loss_sum / len(targets)
    return scores, targets, predictions


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["resnet50", "convnext_tiny", "vit_b_16", "swin_t"])
    parser.add_argument("--augmentation", default="real_only",
                        choices=["real_only", "randaugment", "mixup", "cutmix"])
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--shots", type=int, choices=[5, 10, 20, 50], default=50)
    budget.add_argument("--full-data", action="store_true")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument("--config", type=Path, default=root / "configs/cifar100_training.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this in the AWS project environment.")

    config = json.loads(args.config.read_text())
    split_config = json.loads((root / "configs/cifar100_fewshot.json").read_text())
    manifest_path = root / "configs/splits/cifar100_fewshot.json"
    manifest = json.loads(manifest_path.read_text())
    if args.seed not in split_config["training_seeds"]:
        raise ValueError("The training seed is not in the split configuration.")
    if config["image_size"] != 224 or config["batch_size"] < 2:
        raise ValueError("This development protocol uses 224-pixel inputs and batches of at least two.")
    if config["steps"] < 1 or config["validation_interval"] < 1 or config["warmup_steps"] < 0:
        raise ValueError("Training steps and validation interval must be positive; warmup cannot be negative.")

    train_data = CIFAR100(root / "data/cifar100", train=True, download=False,
                         transform=image_transform(config, args.augmentation == "randaugment"))
    expected = make_splits(train_data.targets, split_config, include_full=args.full_data)
    for key in ("targets_sha256", "split_seed", "validation_per_class", "train_indices", "validation_indices"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"The saved split differs from the dataset or configuration: {key}")
    if manifest["class_names"] != train_data.classes:
        raise ValueError("The class names differ from the saved split.")
    validation_data = CIFAR100(root / "data/cifar100", train=True, download=False,
                              transform=image_transform(config))
    train_ids = expected["full_train_indices"] if args.full_data else manifest["train_indices"][str(args.shots)]
    validation_ids = manifest["validation_indices"]
    steps = config["steps"]
    validation_interval = config["validation_interval"]
    if args.smoke:
        labels = np.asarray(train_data.targets)
        validation_ids = np.asarray(validation_ids)
        validation_ids = sorted(np.concatenate([
            validation_ids[labels[validation_ids] == c][:2] for c in range(100)
        ]).tolist())
        steps, validation_interval = 2, 2
    if len(train_ids) < config["batch_size"]:
        raise ValueError("The batch size exceeds the training subset.")
    steps_per_epoch = len(train_ids) // config["batch_size"]
    resumable = not args.smoke and validation_interval == steps_per_epoch and steps % steps_per_epoch == 0
    if (args.resume or args.stop_after_epochs is not None) and not resumable:
        raise ValueError("Resuming and staged training require whole epochs with validation after each epoch.")
    stop_step = steps
    if args.stop_after_epochs is not None:
        if args.stop_after_epochs < 1:
            raise ValueError("stop-after-epochs must be positive.")
        stop_step = min(steps, args.stop_after_epochs * steps_per_epoch)

    loader_options = {"batch_size": config["batch_size"], "num_workers": config["workers"],
                      "pin_memory": True, "persistent_workers": config["workers"] > 0 and not resumable}
    train_loader = DataLoader(Subset(train_data, train_ids), shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed), **loader_options)
    validation_loader = DataLoader(Subset(validation_data, validation_ids), shuffle=False,
                                   generator=torch.Generator().manual_seed(args.seed + 1), **loader_options)
    batch_augmentation = None
    if args.augmentation == "mixup":
        batch_augmentation = v2.MixUp(alpha=config["mixup_alpha"], num_classes=100)
    elif args.augmentation == "cutmix":
        batch_augmentation = v2.CutMix(alpha=config["cutmix_alpha"], num_classes=100)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = get_model(args.model, weights=None, num_classes=100)
    initial_hash = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(tensor.numpy().tobytes())
    model = model.cuda()
    optimizer_config = config["models"][args.model].copy()
    optimizer_name = optimizer_config.pop("optimizer")
    optimizers = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optimizer = optimizers[optimizer_name](model.parameters(), **optimizer_config)
    mode = "smoke" if args.smoke else "development"
    budget_name = "full" if args.full_data else f"{args.shots}shot"
    output = args.output or root / "runs" / f"{args.model}_{budget_name}_{args.augmentation}_seed{args.seed}_{mode}"
    run_config = {
        "mode": mode, "model": args.model, "augmentation": args.augmentation,
        "shots": None if args.full_data else args.shots, "full_data": args.full_data,
        "seed": args.seed, "weights": None, "all_layers_trainable": True,
        "training": config, "effective_steps": steps, "validation_interval": validation_interval,
        "train_images": len(train_ids), "validation_images": len(validation_ids),
        "steps_per_epoch": len(train_loader),
        "resumable": resumable,
        "epoch_shuffle_seed": "training_seed + zero_based_epoch" if resumable else None,
        "persistent_workers": loader_options["persistent_workers"],
        "train_indices_sha256": hashlib.sha256(np.asarray(train_ids, dtype="<i8").tobytes()).hexdigest(),
        "evaluation_split": "validation_smoke_subset" if args.smoke else "validation",
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "initial_weights_sha256": initial_hash.hexdigest(),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("scripts/train_classifier.py", "scripts/checkpoint.py",
                                       "scripts/metrics.py", "scripts/prepare_cifar100.py")},
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "precision": "float32", "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    step, best_step = 0, 0
    best_key = (-math.inf, -math.inf)
    train_seconds, validation_seconds = 0.0, 0.0
    wall_seconds, previous_peak_memory = 0.0, 0.0
    if args.resume:
        if json.loads((output / "config.json").read_text()) != run_config:
            raise ValueError("Keep the original configuration, code, data split, and environment when resuming.")
        state = load_checkpoint(output / "last.pt", model, optimizer, run_config)
        step, best_step, best_scores = state["step"], state["best_step"], state["best_scores"]
        if step % steps_per_epoch or stop_step <= step:
            raise ValueError("The saved epoch must precede the requested stopping point.")
        best_key = (best_scores["accuracy"], -best_scores["validation_loss"])
        best_targets, best_predictions = state["best_targets"], state["best_predictions"]
        train_seconds, validation_seconds = state["training_seconds"], state["validation_seconds"]
        wall_seconds, previous_peak_memory = state["training_wall_seconds"], state["peak_allocated_mib"]
        torch.save(state.pop("best_model"), output / "best.pt")
        np.savez_compressed(output / "validation_predictions.npz", image_indices=validation_ids,
                            targets=best_targets, predictions=best_predictions)
        (output / "history.jsonl").write_text(state["history"])
        del state
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(f"{args.model} | {args.augmentation} | {mode} | {budget_name} | {steps} planned steps", flush=True)
    print(f"Training: {len(train_ids)} images | validation: {len(validation_ids)} images", flush=True)
    print(f"Starting at step {step}; stopping at step {stop_step}.", flush=True)
    training_loss_sum, training_images = 0.0, 0
    warmup = min(config["warmup_steps"], steps)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    wall_start = training_start = time.perf_counter()
    while step < stop_step:
        if resumable:
            train_loader.generator.manual_seed(args.seed + step // steps_per_epoch)
        for images, labels in train_loader:
            step += 1
            if warmup and step <= warmup:
                multiplier = step / warmup
            else:
                multiplier = 0.5 * (1 + math.cos(math.pi * (step - warmup - 1) / (steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = optimizer_config["lr"] * multiplier
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            if batch_augmentation is not None:
                images, labels = batch_augmentation(images, labels)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != (len(labels), 100):
                raise ValueError("The classifier must output 100 logits per image.")
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            training_loss_sum += loss.detach().item() * len(images)
            training_images += len(images)
            if step % validation_interval == 0 or step == steps:
                torch.cuda.synchronize()
                train_seconds += time.perf_counter() - training_start
                validation_start = time.perf_counter()
                scores, targets, predictions = evaluate(model, validation_loader)
                torch.cuda.synchronize()
                validation_seconds += time.perf_counter() - validation_start
                key = (scores["accuracy"], -scores["validation_loss"])
                if key > best_key:
                    best_key, best_step, best_scores = key, step, scores
                    best_targets, best_predictions = targets, predictions
                    torch.save(model.state_dict(), output / "best.pt")
                    np.savez_compressed(output / "validation_predictions.npz",
                                        image_indices=validation_ids, targets=targets, predictions=predictions)
                with (output / "history.jsonl").open("a") as history:
                    history.write(json.dumps({"step": step, "lr": optimizer.param_groups[0]["lr"],
                                              "training_loss": training_loss_sum / training_images,
                                              "validation": scores}) + "\n")
                print(f"step {step}: training loss {training_loss_sum / training_images:.4f}, "
                      f"validation accuracy {scores['accuracy']:.2f}%, "
                      f"validation loss {scores['validation_loss']:.4f}", flush=True)
                training_loss_sum, training_images = 0.0, 0
                model.train()
                if resumable:
                    wall_seconds += time.perf_counter() - wall_start
                    save_checkpoint(output / "last.pt", model, optimizer, {
                        "run_config": run_config, "step": step, "best_step": best_step,
                        "best_scores": best_scores, "best_targets": best_targets,
                        "best_predictions": best_predictions,
                        "best_model": torch.load(output / "best.pt", map_location="cpu", weights_only=True),
                        "history": (output / "history.jsonl").read_text(),
                        "training_seconds": train_seconds, "validation_seconds": validation_seconds,
                        "training_wall_seconds": wall_seconds,
                        "peak_allocated_mib": max(previous_peak_memory, torch.cuda.max_memory_allocated() / 1024**2),
                    })
                    wall_start = time.perf_counter()
                training_start = time.perf_counter()
            if step == stop_step:
                break

    torch.cuda.synchronize()
    training_wall_seconds = wall_seconds + time.perf_counter() - wall_start
    peak_memory = max(previous_peak_memory, torch.cuda.max_memory_allocated() / 1024**2)
    model.load_state_dict(torch.load(output / "best.pt", map_location="cpu", weights_only=True))
    reloaded_scores, _, reloaded_predictions = evaluate(model, validation_loader)
    with np.load(output / "validation_predictions.npz") as saved:
        np.testing.assert_array_equal(reloaded_predictions, saved["predictions"])
    if not math.isclose(reloaded_scores["validation_loss"], best_scores["validation_loss"], rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Validation loss changed after loading the saved checkpoint.")
    result = {"mode": mode, "best_step": best_step, "validation": best_scores,
              "status": "paused" if step < steps else "complete", "completed_steps": step,
              "planned_steps": steps,
              "training_seconds": train_seconds, "validation_seconds": validation_seconds,
              "training_wall_seconds": training_wall_seconds, "peak_allocated_mib": peak_memory,
              "checkpoint_reload_passed": True}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Checkpoint reload passed. Peak GPU memory: {peak_memory:.0f} MiB. Results: {output}", flush=True)
    if step < steps:
        print("Stopped at the requested epoch. Continue with the same command and --resume, "
              "removing or increasing --stop-after-epochs.", flush=True)
