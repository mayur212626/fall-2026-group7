"""Stable Diffusion img2img pool for CIFAR-100: edits of the real training images (``sd_img2img``).

A third generated set for RQ3, between the two Stage 1 generators: the class-prompt
pool (``sd_prompt``) knows nothing about the real images, the LoRA pools
(``sd_lora``) are adapted to them, and this pool starts from the real images
themselves. Each generated image is one real training image, upsampled to the
pipeline size and partly re-noised and denoised by Stable Diffusion 1.5 with a
class prompt (``--strength``, 0.5 by default: half of the diffusion steps). It is
the image-editing idea of DA-Fusion (Trabucco et al.) and real guidance (He et al.)
without learned class tokens, so it is reported as "SD img2img", not as a
reproduction of either.

Images are generated at the pipeline size and reduced to 32x32, as for the other
pools. Image ``i`` of class ``c`` is an edit of the class's ``i``-th source image,
where the sources are ordered by budget: the images of budget 5 first, then those
that budget 10 adds, then 20, then 50 (each tier in index order). A run with ``k``
real images per class uses the first ``k`` generated images of each class, which
are edits of exactly its own ``k`` real images, so one pool of 50 per class serves
budgets 5 to 50 and the selections are nested. Only training images are read;
validation and test images are never used.

Prompts, negative prompt, seeds, sampler, steps and guidance are those of
``generate_sd``; the image seed is derived from ``(generation seed, c, i)``.
Generation is resumable (one shard per finished class).

Outputs in ``--output-dir``: ``shards/``, ``preview/`` (source and edit side by
side), ``pool.npz``, ``pool.json``, and the ``sd_img2img`` row of the generator
registry.

    python -m src.component.generate_img2img --output-dir data/synthetic/sd15_img2img
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from src.component.generate_sd import (
    MODEL_ID,
    NEGATIVE_PROMPT,
    TEMPLATES,
    class_phrase,
    downsample,
    image_seed,
    merge_shards,
    prompt_for,
    save_shard,
    shard_is_complete,
)
from src.component.generator_registry import REGISTRY, set_row

GENERATOR_ID = "sd_img2img"
SOURCE_BUDGETS = ("5", "10", "20", "50")


def source_order(train_indices: dict[str, list[int]], labels: np.ndarray, num_classes: int,
                 budgets: tuple[str, ...] = SOURCE_BUDGETS) -> dict[int, list[int]]:
    """Training-image indices of every class, ordered so the first ``k`` are budget ``k``'s images.

    ``train_indices`` is the manifest's map from budget to indices. Budgets are
    nested, so each tier adds the images the next budget has and the previous
    one does not, in index order.

    Raises:
        ValueError: If the budgets are not nested or a tier has the wrong size.
    """
    labels = np.asarray(labels)
    order: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    previous: set[int] = set()
    for budget in budgets:
        current = set(train_indices[budget])
        if not previous <= current:
            raise ValueError(f"budget {budget} does not contain the previous budget")
        added = sorted(current - previous)
        for index in added:
            order[int(labels[index])].append(index)
        previous = current
    per_class = int(budgets[-1])
    short = [c for c, v in order.items() if len(v) != per_class]
    if short:
        raise ValueError(f"class {short[0]} has {len(order[short[0]])} source images, expected {per_class}")
    return order


def upsample(image: np.ndarray, size: int) -> Image.Image:
    """A 32x32 uint8 image as a ``size`` x ``size`` PIL image (bicubic), the pipeline's input."""
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB").resize(
        (size, size), Image.Resampling.BICUBIC)


def generate_class(
    pipe: Callable,
    label: int,
    name: str,
    sources: np.ndarray,
    generation_seed: int,
    batch_size: int,
    settings: dict,
    make_generator: Callable[[int], object],
    preview_dir: Path | None = None,
    preview: int = 0,
) -> dict:
    """Edit every source image of one class and return the 32x32 edits with their seeds.

    Args:
        pipe: A diffusers img2img pipeline (or any callable with the same keyword
            arguments that returns an object with ``.images``).
        sources: (n, 32, 32, 3) uint8 real images of the class, in source order.
        settings: ``steps``, ``guidance``, ``strength``, ``size`` and optionally ``negative_prompt``.
        preview_dir, preview: Save the first ``preview`` source/edit pairs side by side as JPEG.
    """
    count = len(sources)
    images = np.empty((count, 32, 32, 3), dtype=np.uint8)
    seeds = np.empty(count, dtype=np.int64)
    for start in range(0, count, batch_size):
        indices = range(start, min(start + batch_size, count))
        batch_seeds = [image_seed(generation_seed, label, i) for i in indices]
        inputs = [upsample(sources[i], settings["size"]) for i in indices]
        output = pipe(
            prompt=[prompt_for(name, i) for i in indices],
            image=inputs,
            strength=settings["strength"],
            negative_prompt=[settings.get("negative_prompt", NEGATIVE_PROMPT)] * len(indices),
            num_inference_steps=settings["steps"],
            guidance_scale=settings["guidance"],
            generator=[make_generator(s) for s in batch_seeds],
        )
        for i, seed, source, image in zip(indices, batch_seeds, inputs, output.images, strict=True):
            images[i] = downsample(image)
            seeds[i] = seed
            if preview_dir is not None and i < preview:
                preview_dir.mkdir(parents=True, exist_ok=True)
                pair = Image.new("RGB", (2 * settings["size"], settings["size"]))
                pair.paste(source, (0, 0))
                pair.paste(image.convert("RGB").resize(source.size), (settings["size"], 0))
                pair.save(preview_dir / f"{label:03d}_{i}.jpg", quality=92)
    return {"images": images, "seeds": seeds, "label": label, "count": count}


def registry_fields(meta: dict) -> dict:
    """Generator-registry values of a finished img2img pool."""
    return {
        "condition": GENERATOR_ID,
        "base_model": meta["model"],
        "revision": meta["revision"],
        "adaptation": f"none; img2img from the real training images, strength {meta['strength']}",
        "adaptation_data": f"edits the first {meta['per_class']} training images per class in budget order "
                           f"({', '.join(meta['source_budgets'])}), so budget k uses edits of its own k images",
        "sampler": f"{meta['scheduler']}, 32x32 upsampled to {meta['size']}x{meta['size']} (bicubic), "
                   f"reduced to {meta['output_size']}x{meta['output_size']} ({meta['resize']})",
        "steps": meta["steps"],
        "guidance": meta["guidance"],
        "prompt_template": f"{' | '.join(meta['templates'])}; negative: {meta['negative_prompt']}",
        "generation_seed": f"{meta['generation_seed']} ({meta['seed_rule']})",
        "images_per_class": meta["per_class"],
        "generation_minutes": round(meta["generation_seconds"] / 60, 1),
        "status": f"generated {meta['finished'][:10]}, pool sha256 {meta['pool_sha256'][:12]}",
    }


def load_pipeline(model: str, revision: str | None, device: str):
    """Load the Stable Diffusion img2img pipeline in fp16 with DPM-Solver++ and no safety checker."""
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
    from huggingface_hub import model_info

    resolved = revision or model_info(model).sha
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model, revision=resolved, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, resolved


def main() -> None:
    """Generate (or resume) the img2img pool, merge it, and record it in the registry."""
    parser = argparse.ArgumentParser(description="Generate the Stable Diffusion img2img pool from the real images.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic/sd15_img2img"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--manifest", type=Path, default=Path("src/component/configs/splits/cifar100.json"))
    parser.add_argument("--classes", type=int, nargs="*", help="only these class labels (for a quick check)")
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.5, help="share of the diffusion applied to the image")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--preview", type=int, default=4, help="source/edit pairs saved per class")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", help="model commit; default: the one recorded for the sd_prompt pool")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()

    import diffusers
    import torch
    import transformers

    from src.component.cifar_splits import labels_sha256
    from src.component.data import load_cifar100, load_manifest

    images, labels = load_cifar100(args.data_root, train=True)
    manifest = load_manifest(args.manifest)
    if labels_sha256(labels) != manifest["labels_sha256"]:
        raise ValueError("CIFAR-100 labels do not match the split manifest")
    from torchvision.datasets import CIFAR100

    class_names = CIFAR100(args.data_root, train=True, download=False).classes
    order = source_order(manifest["train_indices"], np.asarray(labels), len(class_names))
    per_class = int(SOURCE_BUDGETS[-1])
    revision = args.revision
    prompt_meta = Path("data/synthetic/sd15_prompt/pool.json")
    if revision is None and prompt_meta.exists():
        revision = json.loads(prompt_meta.read_text(encoding="utf-8"))["revision"]  # same model as sd_prompt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe, revision = load_pipeline(args.model, revision, device)
    settings = {"steps": args.steps, "guidance": args.guidance, "strength": args.strength, "size": args.size,
                "negative_prompt": NEGATIVE_PROMPT}

    def make_generator(seed: int):
        return torch.Generator(device=device).manual_seed(seed)

    meta_path = args.output_dir / "pool.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    elapsed = previous.get("generation_seconds", 0.0)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for label in (args.classes if args.classes else range(len(class_names))):
        if shard_is_complete(args.output_dir, label, per_class):
            print(f"skip class {label} {class_names[label]} (done)")
            continue
        t0 = time.perf_counter()
        shard = generate_class(pipe, label, class_names[label], images[order[label]], args.generation_seed,
                               args.batch_size, settings, make_generator, args.output_dir / "preview", args.preview)
        save_shard(args.output_dir, shard)
        seconds = time.perf_counter() - t0
        elapsed += seconds
        print(f"class {label:3d} {class_names[label]:<16} {per_class} edits in {seconds / 60:.1f} min", flush=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        progress = {**previous, "started": previous.get("started", started), "generation_seconds": elapsed}
        meta_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

    if args.classes:
        print("partial run (--classes): pool.npz is written only when every class is generated")
        return

    pool_path, sha = merge_shards(args.output_dir, len(class_names), per_class)
    meta = {
        "generator_id": GENERATOR_ID, "model": args.model, "revision": revision,
        "scheduler": type(pipe.scheduler).__name__, "steps": args.steps, "guidance": args.guidance,
        "strength": args.strength, "size": args.size, "output_size": 32, "resize": "PIL bicubic (antialiased)",
        "input_resize": "PIL bicubic 32 to size", "dtype": "float16", "safety_checker": "disabled",
        "per_class": per_class, "source_budgets": list(SOURCE_BUDGETS),
        "source_rule": "class images of budget 5, then those budget 10 adds, then 20, then 50; each tier in index order",
        "source_indices": {str(c): order[c] for c in range(len(class_names))},
        "generation_seed": args.generation_seed,
        "seed_rule": "numpy SeedSequence([generation_seed, label, index]), uint32",
        "templates": list(TEMPLATES), "template_rule": "index mod number of templates",
        "negative_prompt": NEGATIVE_PROMPT,
        "class_phrases": {name: class_phrase(name) for name in class_names},
        "pool": str(pool_path), "pool_sha256": sha, "images": per_class * len(class_names),
        "generation_seconds": elapsed, "started": previous.get("started", started),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "diffusers": diffusers.__version__, "transformers": transformers.__version__,
                     "numpy": np.__version__},
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    set_row(args.registry, GENERATOR_ID, registry_fields(meta))
    print(f"saved {pool_path} ({meta['images']} images, sha256 {sha[:12]}), {elapsed / 3600:.2f} GPU hours")
    print(f"recorded {GENERATOR_ID} in {args.registry}")


if __name__ == "__main__":
    main()
