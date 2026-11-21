"""Stage 3 run list, split evenly across GPUs.

Every Stage 3 dataset is trained on the same grid: ResNet-50 and ViT-B/16,
pretrained and from scratch, every data budget of the dataset, the ten
conditions below and five seeds. The runs are split into ``--parts`` lists of
about equal GPU time, one per instance, without any shared storage: each part
gets every seed's runs in proportion, so all parts finish seeds 0-2 at about
the same time before starting seeds 3 and 4. A run's cost is its step budget
times the measured time per step of its model on an A10G.

    python -m src.component.stage3_jobs --dataset ham10000 --seeds 0 1 2 --parts 3 --summary
    python -m src.component.stage3_jobs --dataset ham10000 --seeds 0 1 2 --parts 3 --part 2

Each printed line is ``init model condition ratio budget seed run_name``;
``run_stage3.sh`` reads them. The split depends only on the arguments, so every
instance computes the same lists. Needs only the standard library.
"""

import argparse
import json
from pathlib import Path

CONFIGS = Path("src/component/configs")
MODELS = ["resnet50", "vit_b_16"]
INITS = ["pretrained", "scratch"]
# (condition, synthetic:real ratio): real-only, six conventional augmentations, and SD class prompts at
# 1:1, with RandAugment at 1:1, and with RandAugment at 5:1 (the best CIFAR-100 condition).
CONDITIONS = [
    ("real_only", 1), ("randaugment", 1), ("trivialaugment", 1), ("autoaugment", 1), ("augmix", 1),
    ("mixup", 1), ("cutmix", 1), ("sd_prompt", 1), ("sd_prompt_randaugment", 1), ("sd_prompt_randaugment", 5),
]
# Seconds per optimizer step at batch 128 and 224x224 on an A10G, from the CIFAR-100 runs
# (ViT-B/16 from scratch at 50 images per class: 8,000 steps in about 62 minutes).
SECONDS_PER_STEP = {"resnet50": 0.24, "vit_b_16": 0.47}


def run_name(init: str, model: str, condition: str, budget: str, seed: int, ratio: int = 1) -> str:
    """The run directory name, as ``train.run_name`` makes it."""
    name = f"{init}_{model}_{condition}_b{budget}_s{seed}"
    return name if ratio == 1 else f"{name}_r{ratio}"


def load_steps(dataset: str, configs: Path = CONFIGS) -> dict[str, dict[str, int]]:
    """Step budget of every data budget, per initialization, from ``train_<init>_<dataset>.json``."""
    return {init: json.loads((Path(configs) / f"train_{init}_{dataset}.json").read_text(encoding="utf-8"))["steps"]
            for init in INITS}


def all_jobs(steps: dict[str, dict[str, int]], seeds: list[int]) -> list[dict]:
    """Every run of the grid for ``seeds``, in seed, initialization, budget, model, condition order."""
    jobs = []
    for seed in seeds:
        for init in INITS:
            for budget in sorted(steps[init], key=int):
                for model in MODELS:
                    for condition, ratio in CONDITIONS:
                        jobs.append({
                            "init": init, "model": model, "condition": condition, "ratio": ratio, "budget": budget,
                            "seed": seed, "name": run_name(init, model, condition, budget, seed, ratio),
                            "hours": steps[init][budget] * SECONDS_PER_STEP[model] / 3600,
                        })
    return jobs


def split(jobs: list[dict], parts: int) -> list[list[dict]]:
    """Split ``jobs`` into ``parts`` lists of about equal hours, seed by seed.

    Within each seed the runs go, longest first, to the part with the least total so far (counting the
    earlier seeds), so the parts stay level after every seed. Each part keeps the grid order.
    """
    if parts < 1:
        raise ValueError("parts must be at least 1")
    position = {job["name"]: i for i, job in enumerate(jobs)}
    load = [0.0] * parts
    assigned: list[list[dict]] = [[] for _ in range(parts)]
    for seed in dict.fromkeys(job["seed"] for job in jobs):
        for job in sorted((j for j in jobs if j["seed"] == seed), key=lambda j: (-j["hours"], position[j["name"]])):
            target = min(range(parts), key=lambda p: (load[p], p))
            assigned[target].append(job)
            load[target] += job["hours"]
    return [sorted(part, key=lambda j: position[j["name"]]) for part in assigned]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["flowers102", "ham10000"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--part", type=int, default=1, help="1-based")
    parser.add_argument("--summary", action="store_true", help="print the estimated hours of every part instead")
    args = parser.parse_args()
    if not 1 <= args.part <= args.parts:
        parser.error("--part must be between 1 and --parts")

    parts = split(all_jobs(load_steps(args.dataset), args.seeds), args.parts)
    if args.summary:
        total = sum(j["hours"] for part in parts for j in part)
        print(f"{args.dataset}, seeds {' '.join(map(str, args.seeds))}: {sum(map(len, parts))} runs, "
              f"~{total:.0f} GPU hours (estimate)")
        for i, part in enumerate(parts, 1):
            print(f"  part {i}: {len(part)} runs, ~{sum(j['hours'] for j in part):.0f} hours")
        return
    for job in parts[args.part - 1]:
        print(job["init"], job["model"], job["condition"], job["ratio"], job["budget"], job["seed"], job["name"])


if __name__ == "__main__":
    main()
