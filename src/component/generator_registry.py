"""Read and update ``src/component/configs/generator_registry.csv``.

The registry has one row per generator: the pretrained Stable Diffusion pool
(``sd_prompt``) and one LoRA adapter per data budget (``lora_sd_b5`` to
``lora_sd_bfull``). ``generate_sd`` fills a row's generation settings from the
pool's ``pool.json`` when the pool is finished, and ``pool_quality`` adds its
FID and Improved Precision & Recall. Other rows and columns are left as they
are.

A finished pool can also be recorded from its ``pool.json`` alone:

    python -m src.component.generator_registry --generator sd_prompt \
        --pool-json data/synthetic/sd15_prompt/pool.json
"""

import argparse
import csv
import json
from pathlib import Path

REGISTRY = Path("src/component/configs/generator_registry.csv")
FIELDS = [
    "generator", "condition", "base_model", "revision", "adaptation", "adaptation_data", "sampler", "steps",
    "guidance", "prompt_template", "generation_seed", "images_per_class", "generation_minutes", "fid",
    "precision", "recall", "quality_file", "status", "ticket",
]


def read_registry(path: Path = REGISTRY) -> list[dict]:
    """Rows of the registry, or no rows if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def set_row(path: Path, generator: str, fields: dict) -> None:
    """Update the row of ``generator`` with ``fields`` (or add it), keeping every other value."""
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown registry columns: {sorted(unknown)}")
    rows = read_registry(path)
    row = next((r for r in rows if r["generator"] == generator), None)
    if row is None:
        row = {name: "" for name in FIELDS} | {"generator": generator}
        rows.append(row)
    row.update({name: "" if value is None else value for name, value in fields.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({name: r.get(name, "") for name in FIELDS} for r in rows)


def generation_fields(meta: dict) -> dict:
    """Registry values for a finished pool, from its ``pool.json``."""
    lora = meta.get("lora")
    prompt = " | ".join(meta["templates"])
    if meta.get("identifier"):
        prompt += f" (class written as '{meta['identifier']} <class>')"
    return {
        "condition": "sd_lora" if lora else "sd_prompt",
        "base_model": meta["model"],
        "revision": meta["revision"],
        "adaptation": (
            f"LoRA rank {lora['rank']} on the U-Net attention projections, {lora['steps']} steps, "
            f"batch {lora['batch_size']}, lr {lora['lr']}, identifier '{lora['identifier']}'" if lora else "none"
        ),
        "adaptation_data": (
            f"training images of budget {lora['budget']} ({lora['train_images']} images)" if lora else "none"
        ),
        "sampler": f"{meta['scheduler']}, {meta['size']}x{meta['size']} reduced to "
                   f"{meta['output_size']}x{meta['output_size']} ({meta['resize']})",
        "steps": meta["steps"],
        "guidance": meta["guidance"],
        "prompt_template": f"{prompt}; negative: {meta['negative_prompt']}",
        "generation_seed": f"{meta['generation_seed']} ({meta['seed_rule']})",
        "images_per_class": meta["per_class"],
        "generation_minutes": round(meta["generation_seconds"] / 60, 1),
        "status": f"generated {meta['finished'][:10]}, pool sha256 {meta['pool_sha256'][:12]}",
    }


def main() -> None:
    """Record a finished pool's settings in the registry."""
    parser = argparse.ArgumentParser(description="Record a generated pool in the generator registry.")
    parser.add_argument("--generator", required=True, help="registry name, e.g. sd_prompt or lora_sd_b5")
    parser.add_argument("--pool-json", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()
    meta = json.loads(args.pool_json.read_text(encoding="utf-8"))
    set_row(args.registry, args.generator, generation_fields(meta))
    print(f"updated {args.generator} in {args.registry}")


if __name__ == "__main__":
    main()
