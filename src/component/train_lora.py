"""LoRA adaptation of Stable Diffusion 1.5 to one CIFAR-100 data budget (condition ``sd_lora``).

Each data budget gets its own adapter, fitted only on that budget's real
training images (proposal, Phase 5), so a generator never sees images the
classifier of that budget does not have. Validation and test images are never
used.

Objective (DreamBooth-style class binding). Every training image is captioned
with a prompt from the same templates as the class-prompt pool, with a rare
identifier token before the class name, e.g. "a photo of a cfr bear.". The
LoRA layers learn what "cfr <class>" looks like in CIFAR-100, and the pools
are generated with the same prompts. The loss is the standard denoising loss:
mean squared error between the added noise and the U-Net's noise prediction.
Only the LoRA layers on the U-Net attention projections (to_q, to_k, to_v,
to_out) are trained; the VAE, text encoder and U-Net weights stay frozen.

Real images are 32x32, so they are upsampled to 512x512 (bicubic) with a
random horizontal flip. The adapter therefore also learns the soft look of
upsampled CIFAR images, which is the look the generated images need before
they are reduced to 32x32.

Every batch, caption, flip, noise sample and timestep depends only on the
training seed and the step. Outputs in ``--output-dir``:

    pytorch_lora_weights.safetensors   the adapter (loaded by generate_sd --lora-dir)
    lora.json                           settings, budget, images, versions, time, loss curve summary
    loss.jsonl                          mean loss every 50 steps

    python -m src.component.train_lora --budget 5 --output-dir data/lora/sd15_lora_b5
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from src.component.generate_sd import IDENTIFIER, MODEL_ID, TEMPLATES, prompt_for

# Optimizer steps per data budget (batch 8). About 16 passes over the 500
# images of the smallest budget, and one pass over the 45,000 images of the
# full training set.
STEPS = {"5": 1000, "10": 1500, "20": 2000, "50": 3000, "full": 6000}
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
LOG_EVERY = 50


def training_plan(num_items: int, batch_size: int, steps: int, seed: int) -> list[dict]:
    """Return, for every step, the item positions, template indices and flips of its batch.

    Items are visited in epochs; epoch ``e`` is a permutation drawn from
    ``(seed, e)``, as in the classifier's sampler, so every image is used
    equally often. Template index and flip of each visit come from
    ``(seed, k)`` for the visit number ``k``.
    """
    plan = []
    epoch, order = -1, np.empty(0, dtype=np.int64)
    for step in range(steps):
        positions, templates, flips = [], [], []
        for k in range(step * batch_size, (step + 1) * batch_size):
            if k // num_items != epoch:
                epoch = k // num_items
                order = np.random.default_rng([seed, epoch]).permutation(num_items)
            rng = np.random.default_rng([seed, 1_000_003, k])
            positions.append(int(order[k % num_items]))
            templates.append(int(rng.integers(len(TEMPLATES))))
            flips.append(bool(rng.integers(2)))
        plan.append({"positions": positions, "templates": templates, "flips": flips})
    return plan


def captions(class_names: Sequence[str], labels: Sequence[int], templates: Sequence[int], identifier: str) -> list[str]:
    """Training captions: the pool's prompt template for each image, with the identifier token."""
    return [prompt_for(class_names[label], t, identifier) for label, t in zip(labels, templates, strict=True)]


def main() -> None:
    """Fit one LoRA adapter on one data budget and save it."""
    parser = argparse.ArgumentParser(description="Fit a Stable Diffusion LoRA adapter to one CIFAR-100 budget.")
    parser.add_argument("--budget", choices=list(STEPS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--manifest", type=Path, default=Path("src/component/configs/splits/cifar100.json"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", help="model commit; default: the current one, recorded in lora.json")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank (planned range 4-16)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, help="override the budget's step count")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--identifier", default=IDENTIFIER)
    args = parser.parse_args()

    out = args.output_dir
    if (out / "pytorch_lora_weights.safetensors").exists():
        print(f"{out} already has an adapter; delete it to fit again")
        return

    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from huggingface_hub import model_info
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from torchvision.datasets import CIFAR100
    from transformers import CLIPTextModel, CLIPTokenizer

    from src.component.cifar_splits import labels_sha256
    from src.component.data import budget_indices, load_manifest

    dataset = CIFAR100(args.data_root, train=True, download=False)
    manifest = load_manifest(args.manifest)
    if labels_sha256(dataset.targets) != manifest["labels_sha256"]:
        raise ValueError("CIFAR-100 labels do not match the split manifest")
    indices = budget_indices(manifest, args.budget)
    images = torch.from_numpy(dataset.data[indices]).permute(0, 3, 1, 2).contiguous()  # (N, 3, 32, 32) uint8
    labels = [dataset.targets[i] for i in indices]
    steps = args.steps or STEPS[args.budget]

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    revision = args.revision or model_info(args.model).sha
    load = {"revision": revision}
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer", **load)
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", **load)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", **load)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet", **load)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler", **load)
    for module in (text_encoder, vae, unet):
        module.requires_grad_(False)
    text_encoder.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float32)  # the SD 1.5 VAE is unstable in fp16
    unet.to(device, dtype=torch.float32)
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=LORA_TARGETS))
    unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))

    plan = training_plan(len(indices), args.batch_size, steps, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    loss_log = (out / "loss.jsonl").open("w", encoding="utf-8")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    unet.train()
    started = time.perf_counter()
    window = []
    for step, batch in enumerate(plan):
        x = images[batch["positions"]].to(device).float() / 255
        flip = torch.tensor(batch["flips"], device=device).view(-1, 1, 1, 1)
        x = torch.where(flip, x.flip(-1), x)
        x = F.interpolate(x, size=(args.resolution, args.resolution), mode="bicubic", align_corners=False)
        x = (x.clamp(0, 1) * 2 - 1)
        prompts = captions(dataset.classes, [labels[p] for p in batch["positions"]], batch["templates"],
                           args.identifier)
        tokens = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            latents = vae.encode(x).latent_dist.sample(generator=generator) * vae.config.scaling_factor
            text = text_encoder(tokens)[0].float()
        noise = torch.randn(latents.shape, device=device, generator=generator)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                          device=device, generator=generator)
        noisy = noise_scheduler.add_noise(latents, noise, t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = unet(noisy, t, text).sample
        loss = F.mse_loss(prediction.float(), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        window.append(loss.item())
        if (step + 1) % LOG_EVERY == 0 or step + 1 == steps:
            record = {"step": step + 1, "loss": float(np.mean(window)), "seconds": time.perf_counter() - started}
            loss_log.write(json.dumps(record) + "\n")
            loss_log.flush()
            print(f"b{args.budget} step {step + 1}/{steps} loss {record['loss']:.4f} "
                  f"({record['seconds'] / 60:.1f} min)", flush=True)
            window = []
    loss_log.close()
    seconds = time.perf_counter() - started

    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(out, unet_lora_layers=lora_state, safe_serialization=True)

    import diffusers
    import peft
    import transformers

    meta = {
        "name": out.name,
        "budget": args.budget,
        "train_images": len(indices),
        "train_indices_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
        "model": args.model,
        "revision": revision,
        "identifier": args.identifier,
        "objective": "DreamBooth-style class binding: identifier token + class name, epsilon-prediction MSE",
        "rank": args.rank,
        "alpha": args.rank,
        "targets": LORA_TARGETS,
        "steps": steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": warmup,
        "weight_decay": 1e-2,
        "grad_clip": 1.0,
        "resolution": args.resolution,
        "augmentation": "bicubic upsampling from 32x32, random horizontal flip",
        "seed": args.seed,
        "final_loss": record["loss"],
        "train_seconds": seconds,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__, "numpy": np.__version__,
        },
        "device": torch.cuda.get_device_name(device),
    }
    (out / "lora.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out} ({steps} steps, {seconds / 60:.1f} min, final loss {record['loss']:.4f})")


if __name__ == "__main__":
    main()
