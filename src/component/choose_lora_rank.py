"""Choose the LoRA rank from the rank pilot (proposal, Phase 5: ranks 4-16).

The pilot fits adapters of rank 4, 8 and 16 on the 50-images-per-class
budget, generates a 1:1 pool from each, and trains ResNet-50 and ViT-B/16 on
real + generated images with pilot seed 100 (not a main training seed). Only
validation scores are used. Runs are read from ``<root>/r<rank>/``.

Rule: the rank with the highest validation accuracy averaged over the two
classifiers, but rank 8 (the middle of the planned range) is kept unless
another rank beats it by more than ``--margin`` percentage points, since a
single pilot seed cannot resolve smaller differences.

    python -m src.component.choose_lora_rank --root runs/pilot-lora --output data/lora/chosen_rank.json
"""

import argparse
import json
import statistics
from pathlib import Path

DEFAULT_RANK = 8


def pilot_scores(root: Path) -> dict[int, dict[str, float]]:
    """Best validation accuracy per rank and model, from ``root/r<rank>/<run>/result.json``."""
    scores: dict[int, dict[str, float]] = {}
    for result_path in sorted(Path(root).glob("r*/*/result.json")):
        rank = int(result_path.parent.parent.name[1:])
        settings = json.loads((result_path.parent / "config.json").read_text(encoding="utf-8"))["settings"]
        accuracy = json.loads(result_path.read_text(encoding="utf-8"))["best_validation"]["accuracy"]
        scores.setdefault(rank, {})[settings["model"]] = accuracy
    return scores


def choose_rank(scores: dict[int, dict[str, float]], margin: float, models: int = 2) -> dict:
    """Apply the selection rule; ranks without a score for every model are ignored.

    Raises:
        ValueError: If no rank, or not the default rank, has complete scores.
    """
    complete = {rank: statistics.mean(by_model.values()) for rank, by_model in scores.items() if len(by_model) == models}
    if DEFAULT_RANK not in complete:
        raise ValueError(f"the pilot has no complete result for rank {DEFAULT_RANK}")
    best = max(complete, key=lambda r: (complete[r], r == DEFAULT_RANK))
    chosen = best if complete[best] - complete[DEFAULT_RANK] > margin else DEFAULT_RANK
    return {"chosen_rank": chosen, "mean_validation_accuracy": complete, "margin": margin,
            "rule": f"highest mean; rank {DEFAULT_RANK} kept unless beaten by more than {margin} points"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose the LoRA rank from the pilot runs.")
    parser.add_argument("--root", type=Path, default=Path("runs/pilot-lora"))
    parser.add_argument("--output", type=Path, default=Path("data/lora/chosen_rank.json"))
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args()
    scores = pilot_scores(args.root)
    decision = {**choose_rank(scores, args.margin), "scores": {str(r): s for r, s in scores.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    for rank in sorted(decision["mean_validation_accuracy"]):
        print(f"rank {rank:>2}: mean validation accuracy {decision['mean_validation_accuracy'][rank]:.2f}  "
              f"{scores[rank]}")
    print(f"chosen rank: {decision['chosen_rank']} ({decision['rule']}) -> {args.output}")


if __name__ == "__main__":
    main()
