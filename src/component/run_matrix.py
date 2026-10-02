"""The Stage 1 run matrix: every planned run with its status and cost.

Stage 1 trains 2 initializations x 2 classifiers x 4 conditions x 5 data
budgets x 3 seeds = 240 runs. Each row names the run directory the runner
writes (``<runs-root>/<stage1-init>/<init>_<model>_<condition>_b<budget>_s<seed>``)
and reads its status from it: done when ``result.json`` exists, running when
only ``config.json`` does, planned otherwise. Regenerate the file after runs
finish:

    python -m src.component.run_matrix --runs-root runs \
        --output src/component/configs/run_matrix.csv

The synthetic conditions are named here as the runner will name them:
``sd_prompt`` uses the pretrained Stable Diffusion pool and ``lora_sd`` the
LoRA adapter of the same budget (generators in ``generator_registry.csv``).
"""

import argparse
import json
import logging
from pathlib import Path

from src.component.analyze_results import write_csv

INITS = ["pretrained", "scratch"]
MODELS = ["resnet50", "vit_b_16"]
CONDITIONS = ["real_only", "randaugment", "sd_prompt", "lora_sd"]
BUDGETS = ["5", "10", "20", "50", "full"]
SEEDS = [0, 1, 2]


def planned_runs() -> list[dict]:
    """One row per Stage 1 run, without status."""
    rows = []
    for init in INITS:
        for model in MODELS:
            for condition in CONDITIONS:
                synthetic = condition in ("sd_prompt", "lora_sd")
                for budget in BUDGETS:
                    generator = {"sd_prompt": "sd_prompt", "lora_sd": f"lora_sd_b{budget}"}.get(condition, "")
                    for seed in SEEDS:
                        rows.append({
                            "research_question": "RQ2, RQ4" if synthetic else "RQ2",
                            "task": "classification",
                            "dataset": "CIFAR-100",
                            "split_manifest": "src/component/configs/splits/cifar100.json",
                            "budget": budget,
                            "init": init,
                            "model": model,
                            "condition": condition,
                            "generator": generator,
                            "synthetic_ratio": "1:1" if synthetic else "0",
                            "seed": seed,
                            "runs_root": f"stage1-{init}",
                            "run": f"{init}_{model}_{condition}_b{budget}_s{seed}",
                        })
    return rows


def run_status(run_dir: Path) -> dict:
    """Status of one run directory, with training minutes and peak GPU memory when done."""
    result_path = Path(run_dir) / "result.json"
    if result_path.exists():
        cost = json.loads(result_path.read_text(encoding="utf-8"))["cost"]
        return {"status": "done", "train_minutes": cost["train_seconds"] / 60, "peak_memory_mb": cost["peak_memory_mb"]}
    status = "running" if (Path(run_dir) / "config.json").exists() else "planned"
    return {"status": status, "train_minutes": "", "peak_memory_mb": ""}


def main() -> None:
    """Write the run matrix with the current status of every run."""
    parser = argparse.ArgumentParser(description="Write the Stage 1 run matrix.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("src/component/configs/run_matrix.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = [{**r, **run_status(args.runs_root / r["runs_root"] / r["run"])} for r in planned_runs()]
    logging.info("saved %s", write_csv(rows, args.output))
    for status in ("done", "running", "planned"):
        logging.info("%s: %d", status, sum(r["status"] == status for r in rows))


if __name__ == "__main__":
    main()
