"""Tests for the one-time test-set evaluation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.component.analyze_results import load_runs
from src.component.data import CifarImages
from src.component.evaluate_test import evaluate_run
from src.component.train import run_training

SETTINGS = {
    "model": "resnet50", "init": "scratch", "condition": "real_only", "budget": "5",
    "seed": 0, "num_classes": 100, "image_size": 32, "batch_size": 4, "steps": 4,
    "evaluations": 2, "warmup_fraction": 0.05, "grad_clip": 1.0, "label_smoothing": 0.1,
    "workers": 0, "recipe": {"optimizer": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
}


def fake_dataset(count: int, seed: int) -> CifarImages:
    """Return random 32x32 images with labels 0-9."""
    images = np.random.default_rng(seed).integers(0, 256, (count, 32, 32, 3), dtype=np.uint8)
    return CifarImages(images, [i % 10 for i in range(count)], list(range(count)), randaugment=False)


class EvaluateRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        self.val_set = fake_dataset(6, seed=2)
        self.result = run_training(SETTINGS, {}, fake_dataset(10, seed=1), self.val_set,
                                   self.run_dir, torch.device("cpu"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_reloaded_checkpoint_reproduces_best_validation_score(self) -> None:
        # Scoring the validation set through evaluate_run must match the run's own result.
        out = evaluate_run(self.run_dir, self.val_set, torch.device("cpu"), "hash")
        self.assertEqual(out["test"]["accuracy"], self.result["best_validation"]["accuracy"])
        self.assertEqual(out["checkpoint_step"], self.result["best_step"])

    def test_writes_test_result_with_predictions(self) -> None:
        evaluate_run(self.run_dir, fake_dataset(8, seed=3), torch.device("cpu"), "abc")
        saved = json.loads((self.run_dir / "test_result.json").read_text())
        self.assertEqual([p["image_id"] for p in saved["test_predictions"]], list(range(8)))
        self.assertEqual(saved["test_labels_sha256"], "abc")
        self.assertIn("macro_f1", saved["test"])

    def test_evaluates_each_run_only_once(self) -> None:
        evaluate_run(self.run_dir, fake_dataset(8, seed=3), torch.device("cpu"), "abc")
        with self.assertRaises(FileExistsError):
            evaluate_run(self.run_dir, fake_dataset(8, seed=3), torch.device("cpu"), "abc")

    def test_refuses_an_incomplete_run(self) -> None:
        (self.run_dir / "result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate_run(self.run_dir, fake_dataset(8, seed=3), torch.device("cpu"), "abc")

    def test_analysis_reads_test_scores(self) -> None:
        out = evaluate_run(self.run_dir, fake_dataset(8, seed=3), torch.device("cpu"), "abc")
        rows = load_runs(Path(self.tmp.name), split="test")
        self.assertEqual(rows[0]["accuracy"], out["test"]["accuracy"])


class LoadRunsSplitTest(unittest.TestCase):
    def test_test_split_skips_runs_without_test_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_training(SETTINGS, {}, fake_dataset(10, seed=1), fake_dataset(6, seed=2),
                         Path(tmp) / "run", torch.device("cpu"))
            self.assertEqual(load_runs(Path(tmp), split="test"), [])
            self.assertEqual(len(load_runs(Path(tmp), split="validation")), 1)


if __name__ == "__main__":
    unittest.main()
