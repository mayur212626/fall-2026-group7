"""Tests for the pilot-run summary."""

import json
import tempfile
import unittest
from pathlib import Path

from src.component.summarize_pilot import plateau_step, summarize

HISTORY = [
    {"step": 10, "val_accuracy": 50.0, "elapsed_seconds": 30.0},
    {"step": 20, "val_accuracy": 60.0, "elapsed_seconds": 60.0},
    {"step": 30, "val_accuracy": 59.8, "elapsed_seconds": 90.0},
    {"step": 40, "val_accuracy": 60.2, "elapsed_seconds": 120.0},
]


class PlateauStepTest(unittest.TestCase):
    def test_first_step_within_tolerance_of_best(self) -> None:
        self.assertEqual(plateau_step(HISTORY, tolerance=0.5), 20)

    def test_zero_tolerance_gives_best_step(self) -> None:
        self.assertEqual(plateau_step(HISTORY, tolerance=0.0), 40)


class SummarizeTest(unittest.TestCase):
    def test_one_row_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "steps" / "pretrained_resnet50_real_only_b5_s100"
            run_dir.mkdir(parents=True)
            settings = {"model": "resnet50", "budget": "5", "steps": 40, "recipe": {"lr": 1e-4}}
            (run_dir / "config.json").write_text(json.dumps({"settings": settings}))
            (run_dir / "history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in HISTORY))

            rows = summarize(Path(tmp), tolerance=0.5)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["model"], row["budget"], row["lr"]), ("resnet50", "5", 1e-4))
        self.assertEqual((row["best_step"], row["best_accuracy"], row["plateau_step"]), (40, 60.2, 20))
        self.assertEqual((row["done_steps"], row["steps"], row["minutes"]), (40, 40, 2.0))


if __name__ == "__main__":
    unittest.main()
