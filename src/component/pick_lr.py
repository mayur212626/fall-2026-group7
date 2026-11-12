"""Choose a model's learning rate from its pilot runs and write it into the training config.

The rule is the one used for ResNet-50 and ViT-B/16: among the finished
real-only pilot runs of the model at one data budget (pilot seed 100,
validation data only), take the learning rate with the highest best
validation accuracy; a tie goes to the lower validation loss, then to the
smaller learning rate. When the chosen rate is the smallest or largest one
tried, a warning says so, since a wider range might do better.

    python -m src.component.pick_lr --root runs/pilot-scratch --model convnext_tiny \
        --config src/component/configs/train_scratch.json

Stage 3 runs choose by balanced accuracy, the metric their checkpoints use:

    python -m src.component.pick_lr --root runs/ham10000-pilot-scratch --model vit_b_16 --budget 20 \
        --metric balanced_accuracy --config src/component/configs/train_scratch_ham10000.json
"""

import argparse
import json
from pathlib import Path


def pilot_results(root: Path, model: str, budget: str, metric: str = "accuracy") -> list[dict]:
    """One row per finished real-only pilot run of ``model`` at ``budget`` below ``root``.

    ``accuracy`` holds the best validation ``metric`` (top-1 accuracy by default).
    """
    rows = []
    for result_path in sorted(Path(root).rglob("result.json")):
        settings = json.loads((result_path.parent / "config.json").read_text(encoding="utf-8"))["settings"]
        if (settings["model"], settings["budget"], settings["condition"]) != (model, budget, "real_only"):
            continue
        best = json.loads(result_path.read_text(encoding="utf-8"))["best_validation"]
        rows.append({"lr": settings["recipe"]["lr"], "accuracy": best[metric], "loss": best["loss"],
                     "run": str(result_path.parent)})
    return sorted(rows, key=lambda r: r["lr"])


def choose(rows: list[dict]) -> dict:
    """The best pilot run: highest accuracy, then lower loss, then smaller learning rate."""
    if not rows:
        raise ValueError("no finished pilot runs")
    return min(rows, key=lambda r: (-r["accuracy"], r["loss"], r["lr"]))


def write_lr(config_path: Path, model: str, lr: float) -> None:
    """Set ``models[model]["lr"]`` in the JSON config, rewriting only that model's line."""
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")
    recipe = json.loads(text)["models"][model]
    recipe["lr"] = lr
    lines = text.splitlines()
    [index] = [i for i, line in enumerate(lines) if line.strip().startswith(f'"{model}":')]
    comma = "," if lines[index].rstrip().endswith(",") else ""
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    lines[index] = f'{indent}"{model}": {json.dumps(recipe)}{comma}'
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget", default="50")
    parser.add_argument("--expect", type=int, default=3, help="number of pilot learning rates that must be finished")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "balanced_accuracy"])
    args = parser.parse_args()

    rows = pilot_results(args.root, args.model, args.budget, args.metric)
    if len(rows) < args.expect:
        raise SystemExit(f"only {len(rows)} of {args.expect} pilot runs are finished in {args.root}")
    best = choose(rows)
    for r in rows:
        mark = "  <- chosen" if r is best else ""
        print(f"  {args.model} b{args.budget} lr {r['lr']:.0e}: val {args.metric.replace('_', ' ')} {r['accuracy']:.2f}, "
              f"loss {r['loss']:.4f}{mark}")
    if best["lr"] in (rows[0]["lr"], rows[-1]["lr"]):
        print(f"  warning: {best['lr']:.0e} is at the edge of the range tried")
    write_lr(args.config, args.model, best["lr"])
    print(f"wrote lr {best['lr']:g} for {args.model} to {args.config}")


if __name__ == "__main__":
    main()
