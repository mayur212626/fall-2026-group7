"""Tests for picking a learning rate from pilot runs (JSON only, no torch)."""

import json
import tempfile
import unittest
from pathlib import Path

from src.component import pick_lr as cp


def pilot_run(root: Path, model: str, lr: float, accuracy: float, loss: float, budget: str = "50",
              condition: str = "real_only") -> None:
    run = root / f"lr_{lr}" / f"scratch_{model}_{condition}_b{budget}_s100"
    run.mkdir(parents=True)
    settings = {"model": model, "budget": budget, "condition": condition, "recipe": {"lr": lr}}
    (run / "config.json").write_text(json.dumps({"settings": settings}))
    (run / "result.json").write_text(json.dumps({"best_validation": {"accuracy": accuracy, "loss": loss}}))


class ChooseTest(unittest.TestCase):
    def test_highest_accuracy_then_lower_loss(self) -> None:
        rows = [{"lr": 1e-3, "accuracy": 20.0, "loss": 3.0}, {"lr": 3e-3, "accuracy": 20.0, "loss": 2.9},
                {"lr": 3e-4, "accuracy": 18.0, "loss": 2.0}]
        self.assertEqual(cp.choose(rows)["lr"], 3e-3)

    def test_reads_only_the_model_budget_and_real_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pilot_run(root, "convnext_tiny", 1e-3, 22.0, 3.1)
            pilot_run(root, "convnext_tiny", 3e-4, 19.0, 3.3)
            pilot_run(root, "resnet50", 1.0, 30.0, 2.5)
            pilot_run(root, "convnext_tiny", 3e-3, 40.0, 2.0, budget="5")
            rows = cp.pilot_results(root, "convnext_tiny", "50")
        self.assertEqual([r["lr"] for r in rows], [3e-4, 1e-3])

    def test_write_lr_changes_only_that_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            source = Path("src/component/configs/train_scratch.json").read_text(encoding="utf-8")
            path.write_text(source)
            cp.write_lr(path, "convnext_tiny", 0.003)
            before, after = json.loads(source), json.loads(path.read_text())
        self.assertEqual(after["models"]["convnext_tiny"]["lr"], 0.003)
        after["models"]["convnext_tiny"]["lr"] = before["models"]["convnext_tiny"]["lr"]
        self.assertEqual(after, before)



class MetricTest(unittest.TestCase):
    def test_stage3_pilots_choose_by_balanced_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for lr, accuracy, balanced in ((1e-4, 80.0, 50.0), (3e-4, 78.0, 60.0)):
                run = root / f"lr_{lr}" / f"scratch_resnet50_real_only_b20_s100"
                run.mkdir(parents=True)
                settings = {"model": "resnet50", "budget": "20", "condition": "real_only", "recipe": {"lr": lr}}
                (run / "config.json").write_text(json.dumps({"settings": settings}))
                (run / "result.json").write_text(json.dumps({"best_validation": {
                    "accuracy": accuracy, "balanced_accuracy": balanced, "loss": 1.0}}))
            self.assertEqual(cp.choose(cp.pilot_results(root, "resnet50", "20"))["lr"], 1e-4)
            self.assertEqual(cp.choose(cp.pilot_results(root, "resnet50", "20", "balanced_accuracy"))["lr"], 3e-4)


if __name__ == "__main__":
    unittest.main()
