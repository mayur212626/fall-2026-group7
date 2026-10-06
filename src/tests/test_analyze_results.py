"""Tests for the results analysis."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.component.analyze_results import (
    aggregate,
    images_per_class,
    latex_table,
    load_runs,
    paired_changes,
    plot_accuracy,
    randaugment_comparison_table,
    series_points,
    write_csv,
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


class WriteCsvTest(unittest.TestCase):
    def test_one_row_per_record_with_lists_joined_by_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.csv"
            write_csv(aggregate(ROWS, "accuracy"), path)
            with path.open(encoding="utf-8", newline="") as f:
                records = list(csv.DictReader(f))
        by_condition = {r["condition"]: r for r in records}
        self.assertEqual(len(records), 2)
        self.assertEqual(by_condition["real_only"]["seeds"], "0 1 2")
        self.assertEqual(float(by_condition["real_only"]["mean"]), 32.0)
        self.assertEqual(float(by_condition["real_only"]["sd"]), 2.0)

    def test_floats_are_rounded_to_four_decimals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv([{"changes": [0.1 + 0.2], "mean": 2 / 3}], Path(tmp) / "t.csv")
            text = path.read_text(encoding="utf-8")
        self.assertEqual(text.splitlines()[1], "0.3,0.6667")


class LatexTableTest(unittest.TestCase):
    def table(self, rows: list[dict], init: str = "pretrained") -> str:
        return latex_table(aggregate(rows, "accuracy"), paired_changes(rows, "randaugment", "real_only", "accuracy"),
                           init=init, metric="accuracy", split="validation")

    def test_row_has_both_conditions_and_the_paired_change(self) -> None:
        self.assertIn(r"ResNet-50 & 5 & 32.00 $\pm$ 2.00 & 37.00 $\pm$ 2.65 & +5.00 $\pm$ 1.00 & gain \\",
                      self.table(ROWS))

    def test_uses_booktabs_and_states_seeds_and_split_in_the_caption(self) -> None:
        table = self.table(ROWS)
        for part in (r"\toprule", r"\midrule", r"\bottomrule", "validation", "3 seeds"):
            self.assertIn(part, table)

    def test_leaves_out_other_initializations(self) -> None:
        scratch = [dict(r, init="scratch", accuracy=r["accuracy"] - 20) for r in ROWS]
        table = self.table(ROWS + scratch)
        self.assertIn("32.00", table)
        self.assertNotIn("12.00", table)


class SyntheticConditionTest(unittest.TestCase):
    rows = ROWS + [row("sd_prompt", 0, 33.0), row("sd_prompt", 1, 31.0), row("sd_prompt", 2, 37.0)]

    def test_paired_change_over_real_only(self) -> None:
        change = paired_changes(self.rows, "sd_prompt", "real_only", "accuracy")[0]
        self.assertEqual(change["changes"], [3.0, -1.0, 3.0])
        self.assertEqual(change["verdict"], "inconclusive")

    def test_latex_table_for_a_synthetic_condition(self) -> None:
        changes = paired_changes(self.rows, "sd_prompt", "real_only", "accuracy")
        table = latex_table(aggregate(self.rows, "accuracy"), changes, init="pretrained", metric="accuracy",
                            split="validation", treatment="sd_prompt")
        self.assertIn("Real-only & SD class prompts & Change", table)
        self.assertIn(r"ResNet-50 & 5 & 32.00 $\pm$ 2.00 & 33.67 $\pm$ 3.06 & +1.67 $\pm$ 2.31 & inconclusive \\", table)
        self.assertIn("tab:stage1-pretrained-accuracy-sd_prompt", table)
        self.assertNotIn("RandAugment", table)


THREE_WAY_ROWS = ROWS + [
    row("sd_prompt", 0, 33.0), row("sd_prompt", 1, 34.0), row("sd_prompt", 2, 35.0),
    row("sd_lora", 0, 31.0), row("sd_lora", 1, 33.0), row("sd_lora", 2, 38.0),
    row("sd_prompt_randaugment", 0, 38.0), row("sd_prompt_randaugment", 1, 39.0), row("sd_prompt_randaugment", 2, 41.0),
]


class RandAugmentComparisonTableTest(unittest.TestCase):
    def table(self, rows: list[dict]) -> str:
        return randaugment_comparison_table(rows, init="pretrained", metric="accuracy", split="validation")

    def test_row_has_each_synthetic_condition_minus_randaugment(self) -> None:
        self.assertIn(r"ResNet-50 & 5 & -3.00 $\pm$ 1.73 $\downarrow$ & -3.00 $\pm$ 1.00 $\downarrow$ & "
                      r"+2.33 $\pm$ 1.15 $\uparrow$ \\", self.table(THREE_WAY_ROWS))

    def test_mixed_signs_are_marked_inconclusive(self) -> None:
        rows = THREE_WAY_ROWS[:-1] + [row("sd_prompt_randaugment", 2, 39.0)]
        self.assertIn(r"$\sim$ \\", self.table(rows))

    def test_missing_conditions_and_budgets_without_synthetic_runs_are_left_out(self) -> None:
        rows = [r for r in THREE_WAY_ROWS if r["condition"] != "sd_lora"]
        rows += [row(c, s, 80.0, budget="full") for c in ("real_only", "randaugment") for s in (0, 1, 2)]
        table = self.table(rows)
        self.assertIn(r"ResNet-50 & 5 & -3.00 $\pm$ 1.73 $\downarrow$ & -- & +2.33", table)
        self.assertNotIn("& full &", table)


class SeriesPointsTest(unittest.TestCase):
    def test_missing_budgets_break_the_line_instead_of_bridging_it(self) -> None:
        rows = THREE_WAY_ROWS + [row("sd_prompt", s, 70.0, budget="50") for s in (0, 1, 2)]
        x, y, _ = series_points(aggregate(rows, "accuracy"), "resnet50", "sd_prompt", ["5", "10", "20", "50"])
        self.assertEqual(x, [5, 10, 20, 50])
        self.assertEqual((y[0], y[3]), (34.0, 70.0))
        self.assertTrue(math.isnan(y[1]) and math.isnan(y[2]))


class PlotAccuracyTest(unittest.TestCase):
    def test_draws_the_synthetic_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            svg = plot_accuracy(aggregate(THREE_WAY_ROWS, "accuracy"), Path(tmp) / "accuracy", init="pretrained")[0]
            text = svg.read_text(encoding="utf-8")
        self.assertIn("ResNet-50, SD class prompts + RandAugment", text)
        self.assertIn("ResNet-50, SD + LoRA", text)

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
