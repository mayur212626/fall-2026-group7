"""Tests for the Stage 1 run matrix."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.component.run_matrix import matrix_rows, planned_runs, run_status


class PlannedRunsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runs = planned_runs()

    def test_stage_1_has_300_distinct_runs(self) -> None:
        self.assertEqual(len(self.runs), 2 * 2 * 5 * 5 * 3)
        self.assertEqual(len({r["run"] for r in self.runs}), 300)

    def test_combined_condition_uses_the_prompt_pool(self) -> None:
        row = next(r for r in self.runs if r["run"] == "pretrained_resnet50_sd_prompt_randaugment_b5_s1")
        self.assertEqual((row["research_question"], row["generator"], row["synthetic_ratio"]), ("RQ2, RQ4", "sd_prompt", "1:1"))

    def test_baseline_row(self) -> None:
        row = next(r for r in self.runs if r["run"] == "pretrained_resnet50_real_only_b5_s0")
        self.assertEqual((row["research_question"], row["generator"], row["synthetic_ratio"]), ("RQ2", "", "0"))
        self.assertEqual(row["runs_root"], "stage1-pretrained")

    def test_lora_rows_use_the_adapter_of_their_budget(self) -> None:
        row = next(r for r in self.runs if r["run"] == "scratch_vit_b_16_sd_lora_b20_s2")
        self.assertEqual((row["research_question"], row["generator"], row["synthetic_ratio"]), ("RQ2, RQ4", "lora_sd_b20", "1:1"))
        self.assertEqual(row["runs_root"], "stage1-scratch")


class GeneratorRegistryTest(unittest.TestCase):
    def test_every_planned_generator_is_registered(self) -> None:
        path = Path(__file__).resolve().parents[1] / "component" / "configs" / "generator_registry.csv"
        with path.open(encoding="utf-8", newline="") as f:
            registered = {row["generator"] for row in csv.DictReader(f)}
        self.assertEqual({r["generator"] for r in planned_runs() if r["generator"]} - registered, set())


class RunStatusTest(unittest.TestCase):
    def test_done_running_and_planned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            done = Path(tmp) / "done"
            done.mkdir()
            (done / "config.json").write_text("{}")
            (done / "result.json").write_text(json.dumps({"cost": {"train_seconds": 90.0, "peak_memory_mb": 512.0}}))
            running = Path(tmp) / "running"
            running.mkdir()
            (running / "config.json").write_text("{}")

            self.assertEqual(run_status(done), {"status": "done", "train_minutes": 1.5, "peak_memory_mb": 512.0})
            self.assertEqual(run_status(running), {"status": "running", "train_minutes": "", "peak_memory_mb": ""})
            self.assertEqual(run_status(Path(tmp) / "missing"),
                             {"status": "planned", "train_minutes": "", "peak_memory_mb": ""})


class KeepMissingTest(unittest.TestCase):
    def test_runs_on_another_machine_keep_their_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            here = Path(tmp) / "stage1-pretrained" / "pretrained_resnet50_sd_prompt_b5_s0"
            here.mkdir(parents=True)
            (here / "config.json").write_text("{}")
            previous = {"pretrained_resnet50_real_only_b5_s0": {"status": "done", "train_minutes": "3.2",
                                                                 "peak_memory_mb": "5877"},
                        "pretrained_resnet50_sd_prompt_b5_s0": {"status": "planned"}}
            rows = {r["run"]: r for r in matrix_rows(Path(tmp), previous)}
            fresh = {r["run"]: r for r in matrix_rows(Path(tmp))}
        self.assertEqual(rows["pretrained_resnet50_real_only_b5_s0"]["status"], "done")
        self.assertEqual(rows["pretrained_resnet50_sd_prompt_b5_s0"]["status"], "running")
        self.assertEqual(fresh["pretrained_resnet50_real_only_b5_s0"]["status"], "planned")


if __name__ == "__main__":
    unittest.main()
