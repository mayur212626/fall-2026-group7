"""Summarize pilot runs to choose step budgets and learning rates.

For every run directory below ``--root`` (any directory with a
``history.jsonl``), prints the best validation accuracy, the step where it
was reached, and the plateau step: the first validation within
``--tolerance`` percentage points of the best. Unfinished runs are included
with the steps done so far.

    python -m src.component.summarize_pilot --root runs/pilot
"""

import argparse
import json
from pathlib import Path


def plateau_step(history: list[dict], tolerance: float) -> int:
    """Return the first step whose validation accuracy is within ``tolerance`` of the best."""
    best = max(record["val_accuracy"] for record in history)
    return next(r["step"] for r in history if r["val_accuracy"] >= best - tolerance)


def summarize(root: Path, tolerance: float) -> list[dict]:
    """Return one summary row per run directory below ``root``, sorted by model, budget and LR."""
    rows = []
    for history_path in sorted(Path(root).rglob("history.jsonl")):
        run_dir = history_path.parent
        lines = history_path.read_text(encoding="utf-8").splitlines()
        history = [json.loads(line) for line in lines if line.strip()]
        if not history:
            continue
        settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"]
        best = max(history, key=lambda r: r["val_accuracy"])
        rows.append({
            "run": str(run_dir.relative_to(root)),
            "model": settings["model"],
            "budget": settings["budget"],
            "lr": settings["recipe"]["lr"],
            "steps": settings["steps"],
            "done_steps": history[-1]["step"],
            "best_step": best["step"],
            "best_accuracy": best["val_accuracy"],
            "plateau_step": plateau_step(history, tolerance),
            "minutes": history[-1]["elapsed_seconds"] / 60,
        })
    budget_order = {"5": 0, "10": 1, "20": 2, "50": 3, "full": 4}
    return sorted(rows, key=lambda r: (r["model"], budget_order.get(r["budget"], 9), r["lr"]))


def main() -> None:
    """Print the summary table."""
    parser = argparse.ArgumentParser(description="Summarize pilot runs.")
    parser.add_argument("--root", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--tolerance", type=float, default=0.5, help="percentage points below the best")
    args = parser.parse_args()

    header = f"{'model':<9} {'budget':>6} {'lr':>8} {'steps':>11} {'best acc':>8} {'best step':>9} {'plateau':>7} {'min':>6}"
    print(header)
    for r in summarize(args.root, args.tolerance):
        progress = f"{r['done_steps']}/{r['steps']}"
        print(
            f"{r['model']:<9} {r['budget']:>6} {r['lr']:>8.0e} {progress:>11} {r['best_accuracy']:>8.2f} "
            f"{r['best_step']:>9} {r['plateau_step']:>7} {r['minutes']:>6.1f}"
        )


if __name__ == "__main__":
    main()
