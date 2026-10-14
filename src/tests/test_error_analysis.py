"""Tests for the error analysis of stored validation predictions."""

import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.component.error_analysis import (
    bootstrap_ci,
    class_recall_change,
    confusion_changes,
    interval_rows,
    interval_table,
    load_predictions,
    paired_runs,
    per_class_recall,
    plot_superclass_change,
    read_superclasses,
    sign_test,
    sign_test_rows,
    superclass_change,
    write_superclasses,
)


def run(condition: str, seed: int, true: list[int], pred: list[int], budget: str = "5") -> dict:
    """Return one run as produced by load_predictions."""
    return {"init": "pretrained", "model": "resnet50", "condition": condition, "budget": budget, "seed": seed,
            "image_ids": np.arange(len(true)), "true": np.array(true), "pred": np.array(pred)}


TRUE = [0, 0, 1, 1]


class LoadPredictionsTest(unittest.TestCase):
    def test_reads_settings_and_orders_predictions_by_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pretrained_resnet50_sd_prompt_b5_s1"
            run_dir.mkdir()
            settings = {"init": "pretrained", "model": "resnet50", "condition": "sd_prompt", "budget": "5", "seed": 1}
            (run_dir / "config.json").write_text(json.dumps({"settings": settings}))
            predictions = [{"image_id": 9, "true": 3, "pred": 4}, {"image_id": 2, "true": 1, "pred": 1}]
            (run_dir / "result.json").write_text(json.dumps({"validation_predictions": predictions}))
            runs = load_predictions(Path(tmp))
        self.assertEqual(len(runs), 1)
        self.assertEqual((runs[0]["condition"], runs[0]["seed"]), ("sd_prompt", 1))
        self.assertEqual(runs[0]["image_ids"].tolist(), [2, 9])
        self.assertEqual(runs[0]["true"].tolist(), [1, 3])
        self.assertEqual(runs[0]["pred"].tolist(), [1, 4])


class PerClassRecallTest(unittest.TestCase):
    def test_recall_in_percent_for_every_class(self) -> None:
        recall = per_class_recall(np.array(TRUE + [2]), np.array([0, 1, 1, 1, 0]), num_classes=3)
        self.assertEqual(recall.tolist(), [50.0, 100.0, 0.0])


class PairedRunsTest(unittest.TestCase):
    def test_pairs_by_seed_and_leaves_out_unpaired_seeds(self) -> None:
        runs = [run("sd_prompt", 0, TRUE, TRUE), run("sd_prompt", 1, TRUE, TRUE), run("real_only", 1, TRUE, TRUE),
                run("real_only", 1, TRUE, TRUE, budget="10")]
        pairs = paired_runs(runs, "pretrained", "resnet50", "5", "sd_prompt", "real_only")
        self.assertEqual([(t["seed"], c["seed"]) for t, c in pairs], [(1, 1)])
        self.assertEqual([c["budget"] for _, c in pairs], ["5"])

    def test_rejects_runs_scored_on_different_images(self) -> None:
        other = run("real_only", 0, TRUE, TRUE)
        other["image_ids"] = np.array([0, 1, 2, 5])
        with self.assertRaises(ValueError):
            paired_runs([run("sd_prompt", 0, TRUE, TRUE), other], "pretrained", "resnet50", "5", "sd_prompt", "real_only")


class ClassRecallChangeTest(unittest.TestCase):
    def test_mean_over_seeds_of_the_paired_recall_change(self) -> None:
        pairs = [(run("sd_prompt", 0, TRUE, [0, 0, 1, 1]), run("real_only", 0, TRUE, [0, 1, 1, 1])),
                 (run("sd_prompt", 1, TRUE, [0, 0, 1, 0]), run("real_only", 1, TRUE, [0, 0, 1, 1]))]
        self.assertEqual(class_recall_change(pairs, num_classes=2).tolist(), [25.0, -25.0])


class SuperclassChangeTest(unittest.TestCase):
    def test_mean_change_of_the_classes_in_each_superclass(self) -> None:
        change = superclass_change(np.array([2.0, 4.0, -6.0, 0.0]), coarse_of=[1, 1, 0, 0], num_superclasses=2)
        self.assertEqual(change.tolist(), [-3.0, 3.0])


class ConfusionChangesTest(unittest.TestCase):
    def test_mean_change_in_errors_per_true_and_predicted_class(self) -> None:
        pairs = [(run("sd_prompt", 0, TRUE, [0, 0, 1, 0]), run("real_only", 0, TRUE, [1, 1, 1, 1])),
                 (run("sd_prompt", 1, TRUE, [0, 1, 1, 0]), run("real_only", 1, TRUE, [0, 1, 1, 1]))]
        changes = {(c["true"], c["pred"]): c for c in confusion_changes(pairs)}
        self.assertEqual(changes[(0, 1)]["change"], -1.0)
        self.assertEqual((changes[(0, 1)]["treatment"], changes[(0, 1)]["control"]), (0.5, 1.5))
        self.assertEqual(changes[(1, 0)]["change"], 1.0)
        self.assertNotIn((0, 0), changes)


class BootstrapCiTest(unittest.TestCase):
    def test_identical_predictions_give_a_zero_interval(self) -> None:
        pairs = [(run("sd_prompt", 0, TRUE, [0, 1, 1, 0]), run("real_only", 0, TRUE, [0, 1, 1, 0]))]
        self.assertEqual(bootstrap_ci(pairs, resamples=200, seed=0), (0.0, 0.0))

    def test_treatment_right_on_every_image_the_control_misses(self) -> None:
        pairs = [(run("sd_prompt", 0, TRUE, TRUE), run("real_only", 0, TRUE, [1, 1, 0, 0]))]
        self.assertEqual(bootstrap_ci(pairs, resamples=200, seed=0), (100.0, 100.0))

    def test_interval_covers_the_observed_change_and_is_reproducible(self) -> None:
        rng = np.random.default_rng(1)
        true = rng.integers(0, 5, 400).tolist()
        treated = [t if rng.random() < 0.6 else (t + 1) % 5 for t in true]
        control = [t if rng.random() < 0.5 else (t + 1) % 5 for t in true]
        pairs = [(run("sd_prompt", 0, true, treated), run("real_only", 0, true, control))]
        observed = 100 * (np.mean(np.array(treated) == true) - np.mean(np.array(control) == true))
        low, high = bootstrap_ci(pairs, resamples=500, seed=0)
        self.assertLess(low, observed)
        self.assertGreater(high, observed)
        self.assertEqual(bootstrap_ci(pairs, resamples=500, seed=0), (low, high))


class SignTestTest(unittest.TestCase):
    def test_five_gains_out_of_five(self) -> None:
        self.assertAlmostEqual(sign_test([1.0, 2.0, 0.5, 3.0, 1.5]), 0.0625)

    def test_zero_changes_are_left_out(self) -> None:
        self.assertAlmostEqual(sign_test([1.0, 2.0, 0.0, 3.0, 1.5, 0.5]), 0.0625)

    def test_one_gain_and_one_loss(self) -> None:
        self.assertEqual(sign_test([1.0, -1.0]), 1.0)


CELL_RUNS = [run("real_only", 0, TRUE, [0, 1, 1, 0]), run("real_only", 1, TRUE, [0, 1, 0, 0]),
             run("sd_prompt", 0, TRUE, [0, 0, 1, 0]), run("sd_prompt", 1, TRUE, [0, 0, 1, 1])]


class IntervalRowsTest(unittest.TestCase):
    def test_paired_change_with_seed_sd_verdict_and_interval_for_compared_conditions(self) -> None:
        rows = interval_rows(CELL_RUNS, "pretrained")
        self.assertEqual([(r["treatment"], r["control"]) for r in rows], [("sd_prompt", "real_only")])
        row = rows[0]
        self.assertEqual((row["seeds"], row["changes"], row["mean"]), ([0, 1], [25.0, 75.0], 50.0))
        self.assertEqual(row["verdict"], "gain")
        self.assertLessEqual(row["ci_low"], row["mean"])
        self.assertGreaterEqual(row["ci_high"], row["mean"])
        self.assertEqual((row["resamples"], row["bootstrap_seed"]), (10_000, 0))

    def test_leaves_out_other_initializations(self) -> None:
        self.assertEqual(interval_rows(CELL_RUNS, "scratch"), [])


class SignTestRowsTest(unittest.TestCase):
    def test_counts_gains_and_losses_per_comparison(self) -> None:
        intervals = [{"init": "scratch", "treatment": "sd_prompt", "control": "randaugment", "mean": m}
                     for m in (-1.0, -2.0, 0.5)]
        self.assertEqual(sign_test_rows(intervals), [{"init": "scratch", "treatment": "sd_prompt", "control": "randaugment",
                                                      "cells": 3, "gains": 1, "losses": 2, "p_value": 1.0}])


class IntervalTableTest(unittest.TestCase):
    def test_cell_shows_mean_and_interval_and_last_row_the_sign_test(self) -> None:
        intervals = interval_rows(CELL_RUNS, "pretrained")
        table = interval_table(intervals, sign_test_rows(intervals), "pretrained")
        r = intervals[0]
        self.assertIn(f"ResNet-50 & 5 & +50.00 [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] & -- & -- \\\\", table)
        self.assertIn(r"\multicolumn{2}{l}{Sign test} & 1/1, $p=1.000$ & -- & --", table)
        self.assertIn("10{,}000 resamples", table)
        self.assertIn(r"\label{tab:stage1-pretrained-bootstrap}", table)


class SuperclassesTest(unittest.TestCase):
    def test_maps_each_fine_class_to_its_superclass_from_the_dataset_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "cifar-100-python"
            folder.mkdir()
            meta = {"fine_label_names": ["apple", "bee", "bus"], "coarse_label_names": ["fruit", "insects", "vehicles"]}
            train = {"fine_labels": [2, 0, 1, 0], "coarse_labels": [2, 0, 1, 0]}
            (folder / "meta").write_bytes(pickle.dumps(meta))
            (folder / "train").write_bytes(pickle.dumps(train))
            path = write_superclasses(Path(tmp), Path(tmp) / "superclasses.csv")
            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            classes = read_superclasses(path)
        self.assertEqual(rows[1], {"class": "1", "class_name": "bee", "superclass": "1", "superclass_name": "insects"})
        self.assertEqual(classes["coarse_of"], [0, 1, 2])
        self.assertEqual(classes["superclass_names"], ["fruit", "insects", "vehicles"])
        self.assertEqual(classes["class_names"][2], "bus")

    def test_rejects_a_class_with_two_superclasses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "cifar-100-python"
            folder.mkdir()
            (folder / "meta").write_bytes(pickle.dumps({"fine_label_names": ["a"], "coarse_label_names": ["x", "y"]}))
            (folder / "train").write_bytes(pickle.dumps({"fine_labels": [0, 0], "coarse_labels": [0, 1]}))
            with self.assertRaises(ValueError):
                write_superclasses(Path(tmp), Path(tmp) / "superclasses.csv")


class PlotSuperclassChangeTest(unittest.TestCase):
    def test_writes_svg_and_pdf_with_both_models_and_sorted_superclasses(self) -> None:
        changes = {"resnet50": np.array([1.0, -2.0, 3.0]), "vit_b_16": np.array([2.0, -1.0, 4.0])}
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_superclass_change(changes, ["fish", "trees", "people"], Path(tmp) / "change",
                                           xlabel="Recall change (points)")
            text = paths[0].read_text(encoding="utf-8")
            self.assertTrue(paths[1].read_bytes().startswith(b"%PDF"))
        self.assertIn("ResNet-50", text)
        self.assertIn("ViT-B/16", text)
        # Largest mean change on top; the y tick labels are written bottom to top.
        self.assertLess(text.index("trees"), text.index("fish"))
        self.assertLess(text.index("fish"), text.index("people"))


if __name__ == "__main__":
    unittest.main()
