"""Tests for the training runner, on tiny random data and the CPU."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.component.data import CifarImages
from src.component.train import (
    is_better,
    load_checkpoint,
    lr_factor,
    make_settings,
    new_state,
    run_training,
    save_checkpoint,
    train_steps,
)

RECIPE = {"optimizer": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4}


def fake_dataset(count: int, randaugment: bool = False) -> CifarImages:
    """Return a dataset of random 32x32 images with labels 0-9."""
    images = np.random.default_rng(1).integers(0, 256, (count, 32, 32, 3), dtype=np.uint8)
    labels = [i % 10 for i in range(count)]
    return CifarImages(images, labels, list(range(count)), randaugment=randaugment)


def settings(**overrides: object) -> dict:
    """Return small run settings that train on the CPU in seconds."""
    base = {
        "model": "resnet50", "init": "scratch", "condition": "real_only", "budget": "5",
        "seed": 0, "num_classes": 100, "image_size": 32, "batch_size": 4, "steps": 4,
        "evaluations": 2, "warmup_fraction": 0.05, "grad_clip": 1.0,
        "label_smoothing": 0.1, "workers": 0, "recipe": RECIPE,
    }
    return {**base, **overrides}


CONFIG = {
    "image_size": 224, "batch_size": 128, "evaluations": 20, "warmup_fraction": 0.05,
    "grad_clip": 1.0, "label_smoothing": 0.1,
    "steps": {"5": 600, "full": None},
    "models": {"resnet50": {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.05}},
}


class MakeSettingsTest(unittest.TestCase):
    def make(self, **overrides: object) -> dict:
        args = {"model": "resnet50", "init": "pretrained", "condition": "real_only",
                "budget": "5", "seed": 0, "workers": 4, **overrides}
        return make_settings(CONFIG, **args)

    def test_takes_recipe_and_steps_from_config(self) -> None:
        run = self.make()
        self.assertEqual(run["steps"], 600)
        self.assertEqual(run["recipe"], CONFIG["models"]["resnet50"])
        self.assertEqual((run["batch_size"], run["num_classes"]), (128, 100))

    def test_steps_override(self) -> None:
        self.assertEqual(self.make(steps=50)["steps"], 50)

    def test_lr_override_changes_only_the_learning_rate(self) -> None:
        run = self.make(lr=3e-4)
        self.assertEqual(run["recipe"], {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.05})
        self.assertEqual(CONFIG["models"]["resnet50"]["lr"], 1e-4)

    def test_rejects_budget_without_steps(self) -> None:
        with self.assertRaises(ValueError):
            self.make(budget="full")


class LrFactorTest(unittest.TestCase):
    def test_warmup_rises_linearly_to_one(self) -> None:
        self.assertAlmostEqual(lr_factor(0, total_steps=100, warmup_steps=5), 1 / 5)
        self.assertAlmostEqual(lr_factor(4, total_steps=100, warmup_steps=5), 1.0)

    def test_cosine_decays_towards_zero(self) -> None:
        factors = [lr_factor(step, 100, 5) for step in range(5, 100)]
        self.assertAlmostEqual(factors[0], 1.0)
        self.assertTrue(all(a > b for a, b in zip(factors, factors[1:])))
        self.assertAlmostEqual(factors[-1], 0.5 * (1 + math.cos(math.pi * 94 / 95)))


class IsBetterTest(unittest.TestCase):
    def test_first_evaluation_is_better_than_none(self) -> None:
        self.assertTrue(is_better({"accuracy": 1.0, "loss": 9.0}, None))

    def test_higher_accuracy_wins(self) -> None:
        self.assertTrue(is_better({"accuracy": 51.0, "loss": 3.0}, {"accuracy": 50.0, "loss": 1.0}))

    def test_equal_accuracy_uses_lower_loss(self) -> None:
        self.assertTrue(is_better({"accuracy": 50.0, "loss": 1.0}, {"accuracy": 50.0, "loss": 2.0}))
        self.assertFalse(is_better({"accuracy": 50.0, "loss": 2.0}, {"accuracy": 50.0, "loss": 1.0}))

    def test_full_tie_keeps_earlier_checkpoint(self) -> None:
        self.assertFalse(is_better({"accuracy": 50.0, "loss": 1.0}, {"accuracy": 50.0, "loss": 1.0}))


class ResumeTest(unittest.TestCase):
    def test_resumed_training_matches_uninterrupted_training(self) -> None:
        run = settings(steps=4)
        dataset = fake_dataset(10, randaugment=True)
        device = torch.device("cpu")

        straight = new_state(run, device)
        train_steps(straight, dataset, run, until_step=4, device=device)

        with tempfile.TemporaryDirectory() as tmp:
            first = new_state(run, device)
            train_steps(first, dataset, run, until_step=2, device=device)
            save_checkpoint(first, Path(tmp) / "last.pt", {"elapsed": 1.0})
            resumed = new_state(run, device)
            extra = load_checkpoint(resumed, Path(tmp) / "last.pt")
        train_steps(resumed, dataset, run, until_step=4, device=device)

        self.assertEqual(extra, {"elapsed": 1.0})
        self.assertEqual(resumed.step, 4)
        for a, b in zip(straight.model.state_dict().values(), resumed.model.state_dict().values()):
            self.assertTrue(torch.equal(a, b))


class RunTrainingResumeTest(unittest.TestCase):
    """A run directory with config.json and last.pt at step 2 of 4, as after a crash."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        self.run_dir.mkdir()
        state = new_state(settings(), torch.device("cpu"))
        train_steps(state, fake_dataset(10), settings(), until_step=2, device=torch.device("cpu"))
        save_checkpoint(state, self.run_dir / "last.pt", {"elapsed": 1.0, "peak_memory_mb": 0.0})
        config = {"settings": settings(), "metadata": {}}
        (self.run_dir / "config.json").write_text(json.dumps(config))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_continues_from_last_checkpoint(self) -> None:
        run_training(settings(), {}, fake_dataset(10), fake_dataset(6), self.run_dir, torch.device("cpu"))
        lines = (self.run_dir / "history.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["step"] for line in lines], [4])
        self.assertTrue((self.run_dir / "result.json").exists())

    def test_refuses_to_resume_with_different_settings(self) -> None:
        with self.assertRaises(ValueError):
            run_training(settings(steps=6), {}, fake_dataset(10), fake_dataset(6), self.run_dir, torch.device("cpu"))


class RunTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        run_training(settings(), {"note": "test"}, fake_dataset(10), fake_dataset(6), self.run_dir, torch.device("cpu"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_logs_one_history_line_per_evaluation(self) -> None:
        lines = (self.run_dir / "history.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["step"] for line in lines], [2, 4])

    def test_saves_result_with_validation_predictions(self) -> None:
        result = json.loads((self.run_dir / "result.json").read_text())
        self.assertIn(result["best_step"], [2, 4])
        self.assertEqual([p["image_id"] for p in result["validation_predictions"]], list(range(6)))
        self.assertIn("accuracy", result["best_validation"])

    def test_keeps_best_checkpoint_and_removes_last(self) -> None:
        self.assertTrue((self.run_dir / "best.pt").exists())
        self.assertFalse((self.run_dir / "last.pt").exists())

    def test_records_settings_and_metadata(self) -> None:
        config = json.loads((self.run_dir / "config.json").read_text())
        self.assertEqual(config["settings"]["steps"], 4)
        self.assertEqual(config["metadata"], {"note": "test"})

    def test_refuses_to_overwrite_a_completed_run(self) -> None:
        with self.assertRaises(FileExistsError):
            run_training(settings(), {}, fake_dataset(10), fake_dataset(6), self.run_dir, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
