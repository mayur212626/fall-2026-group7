"""Tests for the results analysis."""

import json
import tempfile
import unittest
from pathlib import Path

from src.component.analyze_results import (
    aggregate,
    images_per_class,
    load_runs,
    paired_changes,
    plot_accuracy,
)


def row(condition: str, seed: int, accuracy: float, budget: str = "5", model: str = "resnet50") -> dict:
    """Return one run row as produced by load_runs."""
    return {
        "init": "pretrained", "model": model, "condition": condition, "budget": budget,
        "seed": seed, "accuracy": accuracy, "macro_f1": accuracy - 1, "balanced_accuracy": accuracy,
    }


ROWS = [
    row("real_only", 0, 30.0), row("real_only", 1, 32.0), row("real_only", 2, 34.0),
    row("randaugment", 0, 35.0), row("randaugment", 1, 36.0), row("randaugment", 2, 40.0),
]


class LoadRunsTest(unittest.TestCase):
    def test_reads_settings_and_best_validation_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pretrained_resnet50_real_only_b5_s0"
            run_dir.mkdir()
            settings = {"init": "pretrained", "model": "resnet50", "condition": "real_only",
                        "budget": "5", "seed": 0}
            (run_dir / "config.json").write_text(json.dumps({"settings": settings}))
            result = {"best_step": 540, "cost": {"train_seconds": 120.0},
                      "best_validation": {"accuracy": 35.5, "macro_f1": 34.0, "balanced_accuracy": 35.5}}
            (run_dir / "result.json").write_text(json.dumps(result))
            rows = load_runs(Path(tmp))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["accuracy"], 35.5)
        self.assertEqual((rows[0]["condition"], rows[0]["seed"], rows[0]["best_step"]), ("real_only", 0, 540))
        self.assertEqual(rows[0]["train_minutes"], 2.0)


class AggregateTest(unittest.TestCase):
    def test_mean_and_sample_standard_deviation_per_condition(self) -> None:
        groups = {g["condition"]: g for g in aggregate(ROWS, "accuracy")}
        self.assertEqual(groups["real_only"]["mean"], 32.0)
        self.assertEqual(groups["real_only"]["sd"], 2.0)
        self.assertEqual(groups["real_only"]["n"], 3)

    def test_single_seed_has_no_standard_deviation(self) -> None:
        groups = aggregate([row("real_only", 0, 30.0)], "accuracy")
        self.assertIsNone(groups[0]["sd"])


class PairedChangesTest(unittest.TestCase):
    def test_changes_are_paired_by_seed(self) -> None:
        change = paired_changes(ROWS, "randaugment", "real_only", "accuracy")[0]
        self.assertEqual(change["changes"], [5.0, 4.0, 6.0])
        self.assertEqual(change["mean"], 5.0)
        self.assertEqual(change["verdict"], "gain")

    def test_mixed_signs_are_inconclusive(self) -> None:
        rows = [row("real_only", 0, 30.0), row("real_only", 1, 30.0),
                row("randaugment", 0, 31.0), row("randaugment", 1, 29.0)]
        self.assertEqual(paired_changes(rows, "randaugment", "real_only", "accuracy")[0]["verdict"], "inconclusive")

    def test_all_negative_is_a_loss(self) -> None:
        rows = [row("real_only", 0, 30.0), row("randaugment", 0, 28.0)]
        self.assertEqual(paired_changes(rows, "randaugment", "real_only", "accuracy")[0]["verdict"], "loss")

    def test_seeds_without_a_pair_are_left_out(self) -> None:
        rows = ROWS + [row("randaugment", 3, 50.0)]
        self.assertEqual(paired_changes(rows, "randaugment", "real_only", "accuracy")[0]["seeds"], [0, 1, 2])


class ImagesPerClassTest(unittest.TestCase):
    def test_full_budget_is_450_images_per_class(self) -> None:
        self.assertEqual(images_per_class("full"), 450)
        self.assertEqual(images_per_class("20"), 20)


class PlotAccuracyTest(unittest.TestCase):
    def test_writes_svg_and_pdf(self) -> None:
        rows = ROWS + [row(c, s, a + 10, budget="full") for c, s, a in
                       [("real_only", 0, 80.0), ("real_only", 1, 81.0), ("randaugment", 0, 82.0), ("randaugment", 1, 83.0)]]
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_accuracy(aggregate(rows, "accuracy"), Path(tmp) / "accuracy", init="pretrained")
            self.assertEqual(sorted(p.suffix for p in paths), [".pdf", ".svg"])
            self.assertTrue(paths[0].read_text(encoding="utf-8").lstrip().startswith("<?xml"))
            self.assertTrue(paths[1].read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
