"""Stable Diffusion image pools for CIFAR-100: class prompts (``sd_prompt``) and LoRA (``sd_lora``).

Stable Diffusion 1.5 is used as released, without any training on CIFAR-100.
Each image comes from a text prompt built from its class name and one of a
fixed list of prompt templates, and from its own seed. Images are generated
at 512x512 and reduced to 32x32, the resolution of the real images, so a
classifier cannot tell real from generated images by sharpness (proposal,
Section 2).

The pool holds ``--per-class`` images per class (450 by default: enough for a
1:1 synthetic-to-real ratio with the full training set). A run with ``k``
real images per class uses the first ``k`` generated images of each class,
so one pool serves every data budget and the selections are nested.

Everything that decides an image is recorded: model and revision, scheduler,
steps, guidance, prompt, negative prompt and seed. Image ``i`` of class ``c``
always gets the template ``i mod len(TEMPLATES)`` and the seed derived from
``(generation seed, c, i)``, so a pool can be regenerated image by image.
Generation is resumable: each class is saved as a shard when it finishes and
finished classes are skipped.

Outputs in ``--output-dir``:

    shards/class_XXX.npz  one file per finished class
    preview/XXX_i.jpg     the first ``--preview`` full-size images per class
    pool.npz              the merged pool (see ``synthetic_pool``)
    pool.json             settings, versions, class prompts and the pool SHA-256

and the generator's row in ``src/component/configs/generator_registry.csv``
(see ``generator_registry``).

With ``--lora-dir`` the pipeline first loads a LoRA adapter from
``train_lora`` and every prompt names the class with the adapter's identifier
token (``"a photo of a cfr bear."``), which is the ``sd_lora`` condition.

    python -m src.component.generate_sd --output-dir data/synthetic/sd15_prompt
    python -m src.component.generate_sd --output-dir data/synthetic/sd15_lora_b5 \
        --lora-dir data/lora/sd15_lora_b5 --per-class 5 --generator-id lora_sd_b5
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

from src.component.generator_registry import REGISTRY, generation_fields, set_row
from src.component.synthetic_pool import save_pool

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
GENERATOR_ID = "sd_prompt"
TEMPLATES = (
    "a photo of {article} {name}.",
    "a close-up photo of {article} {name}.",
    "a photo of {article} {name} in its natural setting.",
    "a bright photo of {article} {name}.",
    "a photo of a small {name}.",
    "a photo of a large {name}.",
    "a cropped photo of {article} {name}.",
    "a good photo of {article} {name}.",
)
NEGATIVE_PROMPT = "text, watermark, logo, cartoon, drawing, illustration, painting, blurry, low quality"
# With a LoRA adapter the target is the look of the real CIFAR photos, which are small and soft,
# so the negative prompt keeps only artefacts that are never wanted.
LORA_NEGATIVE_PROMPT = "text, watermark, logo"
IDENTIFIER = "cfr"
# CIFAR-100 class names that are ambiguous as plain English prompts.
CLASS_PHRASES = {
    "aquarium_fish": "aquarium fish",
    "cattle": "cow",
    "keyboard": "computer keyboard",
    "mouse": "field mouse",
    "plain": "grassy plain landscape",
    "ray": "stingray",
    "seal": "harbor seal",
    "sweet_pepper": "sweet pepper",
    "tank": "military tank",
}
OUTPUT_SIZE = 32


def class_phrase(name: str) -> str:
    """Return the text used for a CIFAR-100 class name in prompts."""
    return CLASS_PHRASES.get(name, name.replace("_", " "))


def prompt_for(name: str, index: int, identifier: str | None = None) -> str:
    """Return the prompt of image ``index`` of class ``name``; templates repeat in order.

    With an ``identifier`` (the LoRA token), the class is written as
    ``"<identifier> <class>"``, e.g. ``"a photo of a cfr bear."``.
    """
    phrase = class_phrase(name)
    if identifier:
        phrase = f"{identifier} {phrase}"
    article = "an" if phrase[0] in "aeiou" else "a"
    return TEMPLATES[index % len(TEMPLATES)].format(article=article, name=phrase)


def image_seed(generation_seed: int, label: int, index: int) -> int:
    """Seed of image ``index`` of class ``label``; distinct and fixed for every image."""
    return int(np.random.SeedSequence([generation_seed, label, index]).generate_state(1, np.uint32)[0])


def downsample(image: Image.Image, size: int = OUTPUT_SIZE) -> np.ndarray:
    """Reduce a generated image to ``size`` x ``size`` RGB uint8 with antialiased bicubic resampling."""
    small = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(small, dtype=np.uint8)


def generate_class(
    pipe: Callable,
    label: int,
    name: str,
    per_class: int,
    generation_seed: int,
    batch_size: int,
    settings: dict,
    make_generator: Callable[[int], object],
    preview_dir: Path | None = None,
    preview: int = 0,
    identifier: str | None = None,
) -> dict:
    """Generate ``per_class`` images of one class and return them with their seeds.

    Args:
        pipe: A diffusers text-to-image pipeline (or any callable with the same
            keyword arguments that returns an object with ``.images``).
        make_generator: Builds a seeded random generator from an image seed.
        settings: ``steps``, ``guidance``, ``size`` and optionally ``negative_prompt``.
        identifier: LoRA identifier token added before the class name.
        preview_dir, preview: Save the first ``preview`` full-size images as JPEG.
    """
    images = np.empty((per_class, OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)
    seeds = np.empty(per_class, dtype=np.int64)
    for start in range(0, per_class, batch_size):
        indices = range(start, min(start + batch_size, per_class))
        batch_seeds = [image_seed(generation_seed, label, i) for i in indices]
        output = pipe(
            prompt=[prompt_for(name, i, identifier) for i in indices],
            negative_prompt=[settings.get("negative_prompt", NEGATIVE_PROMPT)] * len(indices),
            num_inference_steps=settings["steps"],
            guidance_scale=settings["guidance"],
            height=settings["size"],
            width=settings["size"],
            generator=[make_generator(s) for s in batch_seeds],
        )
        for i, seed, image in zip(indices, batch_seeds, output.images, strict=True):
            images[i] = downsample(image)
            seeds[i] = seed
            if preview_dir is not None and i < preview:
                preview_dir.mkdir(parents=True, exist_ok=True)
                image.convert("RGB").save(preview_dir / f"{label:03d}_{i}.jpg", quality=92)
    return {"images": images, "seeds": seeds, "label": label, "count": per_class}


def shard_path(output_dir: Path, label: int) -> Path:
    return Path(output_dir) / "shards" / f"class_{label:03d}.npz"


def save_shard(output_dir: Path, shard: dict) -> Path:
    """Save one finished class; written to a temporary file first."""
    path = shard_path(output_dir, shard["label"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, images=shard["images"], seeds=shard["seeds"], label=shard["label"], count=shard["count"])
    tmp.replace(path)
    return path


def shard_is_complete(output_dir: Path, label: int, per_class: int) -> bool:
    path = shard_path(output_dir, label)
    if not path.exists():
        return False
    with np.load(path) as data:
        return int(data["count"]) >= per_class


def merge_shards(output_dir: Path, num_classes: int, per_class: int) -> tuple[Path, str]:
    """Combine the class shards into ``pool.npz`` and return its path and SHA-256.

    Raises:
        FileNotFoundError: If a class has no complete shard.
    """
    images, labels, index = [], [], []
    for label in range(num_classes):
        if not shard_is_complete(output_dir, label, per_class):
            raise FileNotFoundError(f"class {label} has no complete shard in {output_dir}")
        with np.load(shard_path(output_dir, label)) as data:
            images.append(data["images"][:per_class])
        labels.append(np.full(per_class, label, dtype=np.int64))
        index.append(np.arange(per_class, dtype=np.int64))
    path = Path(output_dir) / "pool.npz"
    sha = save_pool(path, np.concatenate(images), np.concatenate(labels), np.concatenate(index))
    return path, sha


def load_pipeline(model: str, revision: str | None, device: str):
    """Load Stable Diffusion in fp16 with the DPM-Solver++ scheduler and no safety checker.

    The safety checker is off because it replaces any flagged image with a
    black one, which would silently leave black images in the pool.
    Returns the pipeline and the resolved model revision.
    """
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from huggingface_hub import model_info

    resolved = revision or model_info(model).sha
    pipe = StableDiffusionPipeline.from_pretrained(
        model, revision=resolved, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, resolved


def main() -> None:
    """Generate (or resume) the pool, merge it, and record it in the registry."""
    parser = argparse.ArgumentParser(description="Generate the Stable Diffusion class-prompt pool.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic/sd15_prompt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--per-class", type=int, default=450)
    parser.add_argument("--classes", type=int, nargs="*", help="only these class labels (for a quick check)")
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--preview", type=int, default=4, help="full-size JPEGs saved per class")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", help="model commit; default: the current one, recorded in pool.json")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--lora-dir", type=Path, help="LoRA adapter from train_lora (sd_lora condition)")
    parser.add_argument("--generator-id", help="registry name: sd_prompt (default) or lora_sd_b<budget>")
    args = parser.parse_args()

    import diffusers
    import torch
    import transformers
    from torchvision.datasets import CIFAR100

    class_names = CIFAR100(args.data_root, train=True, download=False).classes
    labels = args.classes if args.classes else list(range(len(class_names)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe, revision = load_pipeline(args.model, args.revision, device)
    lora = None
    identifier = None
    negative_prompt = NEGATIVE_PROMPT
    if args.lora_dir:
        lora = json.loads((args.lora_dir / "lora.json").read_text(encoding="utf-8"))
        if lora["revision"] != revision:
            raise ValueError(f"adapter was trained on revision {lora['revision']}, pipeline has {revision}")
        pipe.load_lora_weights(str(args.lora_dir))
        identifier = lora["identifier"]
        negative_prompt = LORA_NEGATIVE_PROMPT
    generator_id = args.generator_id or (f"lora_sd_b{lora['budget']}" if lora else GENERATOR_ID)
    settings = {"steps": args.steps, "guidance": args.guidance, "size": args.size, "negative_prompt": negative_prompt}

    def make_generator(seed: int):
        return torch.Generator(device=device).manual_seed(seed)

    meta_path = args.output_dir / "pool.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    elapsed = previous.get("generation_seconds", 0.0)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for label in labels:
        if shard_is_complete(args.output_dir, label, args.per_class):
            print(f"skip class {label} {class_names[label]} (done)")
            continue
        t0 = time.perf_counter()
        shard = generate_class(
            pipe, label, class_names[label], args.per_class, args.generation_seed, args.batch_size,
            settings, make_generator, args.output_dir / "preview", args.preview, identifier,
        )
        save_shard(args.output_dir, shard)
        seconds = time.perf_counter() - t0
        elapsed += seconds
        print(f"class {label:3d} {class_names[label]:<16} {args.per_class} images in {seconds / 60:.1f} min "
              f"({args.per_class / seconds:.2f} img/s)", flush=True)
        # Record progress after every class so generation time survives a restart.
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({**previous, "generation_seconds": elapsed}, indent=2) + "\n", encoding="utf-8")

    if args.classes:
        print("partial run (--classes): pool.npz is written only when every class is generated")
        return

    pool_path, sha = merge_shards(args.output_dir, len(class_names), args.per_class)
    meta = {
        "generator_id": generator_id,
        "model": args.model,
        "revision": revision,
        "scheduler": type(pipe.scheduler).__name__,
        "steps": args.steps,
        "guidance": args.guidance,
        "size": args.size,
        "output_size": OUTPUT_SIZE,
        "resize": "PIL bicubic (antialiased)",
        "dtype": "float16",
        "safety_checker": "disabled",
        "per_class": args.per_class,
        "generation_seed": args.generation_seed,
        "seed_rule": "numpy SeedSequence([generation_seed, label, index]), uint32",
        "templates": list(TEMPLATES),
        "template_rule": "index mod number of templates",
        "identifier": identifier,
        "lora": lora,
        "negative_prompt": negative_prompt,
        "class_phrases": {name: class_phrase(name) for name in class_names},
        "pool": str(pool_path),
        "pool_sha256": sha,
        "images": args.per_class * len(class_names),
        "generation_seconds": elapsed,
        "started": previous.get("started", started),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "diffusers": diffusers.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    set_row(args.registry, generator_id, generation_fields(meta))
    print(f"saved {pool_path} ({meta['images']} images, sha256 {sha[:12]}), {elapsed / 3600:.2f} GPU hours")
    print(f"recorded {generator_id} in {args.registry}")


if __name__ == "__main__":
    main()
