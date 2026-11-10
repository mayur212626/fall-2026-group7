"""Tests for the Stage 2 results (run folders made of JSON files, no torch)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.component import analyze_results as ar
from src.component import stage2_results as s2


def make_run(root: Path, init: str, model: str, condition: str, budget: str, seed: int, accuracy: float,
             ratio: int = 1, test: float | None = None) -> None:
    name = f"{init}_{model}_{condition}_b{budget}_s{seed}" + (f"_r{ratio}" if ratio != 1 else "")
    run = root / name
    run.mkdir(parents=True)
    settings = {"init": init, "model": model, "condition": condition, "budget": budget, "seed": seed}
    if condition.startswith("sd_"):
        settings["synthetic"] = {"ratio": ratio}
    scores = {"accuracy": accuracy, "macro_f1": accuracy, "balanced_accuracy": accuracy}
    (run / "config.json").write_text(json.dumps({"settings": settings}))
    (run / "result.json").write_text(json.dumps({"best_validation": scores, "best_step": 10,
                                                 "cost": {"train_seconds": 60}}))
    if test is not None:
        (run / "test_result.json").write_text(json.dumps({"test": {**scores, "accuracy": test}}))


class LoadTest(unittest.TestCase):
    def test_ratio_runs_get_their_own_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            make_run(Path(tmp), "scratch", "resnet50", "sd_prompt", "5", 0, 10.0)
            make_run(Path(tmp), "scratch", "resnet50", "sd_prompt", "5", 0, 12.0, ratio=2)
            make_run(Path(tmp), "scratch", "resnet50", "real_only", "5", 0, 9.0)
            conditions = sorted(r["condition"] for r in ar.load_runs(Path(tmp)))
        self.assertEqual(conditions, ["real_only", "sd_prompt", "sd_prompt_r2"])

    def test_runs_from_several_folders_are_combined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            make_run(Path(tmp) / "stage1-pretrained", "pretrained", "resnet50", "real_only", "5", 0, 40.0)
            make_run(Path(tmp) / "stage2-mixing-pretrained", "pretrained", "resnet50", "mixup", "5", 0, 43.0)
            rows = s2.load("pretrained", "validation", [f"{tmp}/stage1-{{init}}", f"{tmp}/stage2-mixing-{{init}}",
                                                        f"{tmp}/missing-{{init}}"])
        self.assertEqual(sorted(r["condition"] for r in rows), ["mixup", "real_only"])


class StudyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = []
        for seed, (real, ra, sd, mixup, r2) in enumerate([(10, 15, 11, 13, 9), (11, 16, 12, 14, 10),
                                                          (12, 17, 13, 15, 11)]):
            for condition, acc in (("real_only", real), ("randaugment", ra), ("sd_prompt", sd), ("mixup", mixup),
                                   ("sd_prompt_r2", r2)):
                self.rows.append({"init": "scratch", "model": "resnet50", "condition": condition, "budget": "5",
                                  "seed": seed, "accuracy": float(acc)})
            # a budget without Stage 2 runs is left out of the Stage 2 tables
            for condition, acc in (("real_only", 30.0), ("sd_prompt", 31.0), ("randaugment", 32.0)):
                self.rows.append({"init": "scratch", "model": "resnet50", "condition": condition, "budget": "20",
                                  "seed": seed, "accuracy": acc})

    def test_mixing_comparisons_and_verdicts(self) -> None:
        changes = {c["label"]: c for c in s2.study_changes(self.rows, "mixing", "accuracy")}
        self.assertEqual(set(changes), {"RA$-$real", "Mixup$-$real", "SD$-$RA", "SD$-$Mixup"})  # no CutMix runs
        self.assertAlmostEqual(changes["SD$-$Mixup"]["mean"], -2.0)
        self.assertEqual(changes["SD$-$Mixup"]["verdict"], "loss")
        self.assertEqual({c["budget"] for c in changes.values()}, {"5"})

    def test_ratio_table_marks_missing_comparisons(self) -> None:
        changes = s2.study_changes(self.rows, "ratio", "accuracy")
        self.assertAlmostEqual({c["label"]: c for c in changes}["2:1$-$real"]["mean"], -1.0)
        table = s2.study_table(changes, "ratio", "scratch", "accuracy", "validation")
        self.assertIn(r"\label{tab:stage2-ratio-scratch}", table)
        self.assertIn(r"ResNet-50 & 5 & +1.00 $\pm$ 0.00 $\uparrow$ & -1.00 $\pm$ 0.00 $\downarrow$ & --", table)


class TestLabelTest(unittest.TestCase):
    def test_test_split_tables_get_their_own_labels(self) -> None:
        rows = [{"init": "pretrained", "model": "resnet50", "condition": c, "budget": "5", "seed": s,
                 "accuracy": a + s} for s in range(3) for c, a in (("real_only", 40.0), ("randaugment", 42.0),
                                                                   ("sd_prompt", 50.0))]
        groups = ar.aggregate(rows, "accuracy")
        changes = ar.paired_changes(rows, "sd_prompt", "real_only", "accuracy")
        test = ar.latex_table(groups, changes, "pretrained", "accuracy", "test", "sd_prompt")
        validation = ar.latex_table(groups, changes, "pretrained", "accuracy", "validation", "sd_prompt")
        self.assertIn(r"\label{tab:stage1-pretrained-accuracy-sd_prompt-test}", test)
        self.assertIn(r"\label{tab:stage1-pretrained-accuracy-sd_prompt}", validation)
        self.assertIn(r"\label{tab:stage1-pretrained-accuracy-vs-randaugment-test}",
                      ar.randaugment_comparison_table(rows, "pretrained", "accuracy", "test"))


class EndToEndTest(unittest.TestCase):
    def test_main_writes_tables_and_the_ratio_figure_for_the_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage1, ratio = Path(tmp) / "stage1-pretrained", Path(tmp) / "stage2-ratio-pretrained"
            for seed in range(3):
                for condition, acc in (("real_only", 40), ("randaugment", 42), ("sd_prompt", 50)):
                    make_run(stage1, "pretrained", "vit_b_16", condition, "5", seed, acc, test=acc - 1.0)
                for r, acc in ((2, 53), (5, 55)):
                    make_run(ratio / f"r{r}", "pretrained", "vit_b_16", "sd_prompt", "5", seed, acc, ratio=r,
                             test=acc - 1.0)
            tables, figures = Path(tmp) / "tables", Path(tmp) / "fig"
            argv = ["stage2_results", "--split", "test", "--tables", str(tables), "--figures", str(figures),
                    "--roots", f"{tmp}/stage1-{{init}}", f"{tmp}/stage2-ratio-{{init}}"]
            with mock.patch.object(sys, "argv", argv):
                s2.main()
            with (tables / "stage2_paired_test.csv").open(encoding="utf-8") as f:
                paired = {r["label"]: r for r in ar.csv.DictReader(f)}
            self.assertAlmostEqual(float(paired["5:1$-$real"]["mean"]), 15.0)
            self.assertTrue((tables / "stage2_ratio_pretrained_test.tex").exists())
            self.assertFalse((tables / "stage2_convnext_pretrained_test.tex").exists())
            self.assertTrue((figures / "stage2_ratio_gain_test.pdf").exists())


if __name__ == "__main__":
    unittest.main()
