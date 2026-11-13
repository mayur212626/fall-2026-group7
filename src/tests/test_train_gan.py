"""Tests for the StyleGAN2-ADA training loop (CPU, tiny networks, 8x8 images)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.component.train_gan import (
    batch_positions,
    ema_beta,
    lazy_regularization,
    make_settings,
    run_training,
)


def tiny_settings(**overrides) -> dict:
    settings = make_settings(budget="50", seed=0, kimg=0.024, snapshot_kimg=0.008)
    settings.update({
        "num_classes": 3, "resolution": 8, "channels": 8, "z_dim": 8, "w_dim": 8, "d_mapping_layers": 2,
        "mbstd_group": 2, "batch_size": 4, "r1_interval": 2, "fid_per_class": 2, "ema_kimg": 0.01,
    })
    settings.update(overrides)
    return settings


def tiny_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (12, 8, 8, 3), dtype=np.uint8), np.repeat(np.arange(3), 4)


def mean_pixel_fid(images: np.ndarray) -> float:
    """Stand-in for FID: any number that depends on the images."""
    return float(images.mean())


class ClassifierConditionTest(unittest.TestCase):
    def test_the_gan_pool_is_added_to_the_real_images_with_no_other_augmentation(self) -> None:
        from src.component.train import AUGMENTED_CONDITIONS, SYNTHETIC_CONDITIONS, SYNTHETIC_ONLY_CONDITIONS
        self.assertIn("gan", SYNTHETIC_CONDITIONS)
        self.assertNotIn("gan", SYNTHETIC_ONLY_CONDITIONS)
        self.assertNotIn("gan", AUGMENTED_CONDITIONS)


class HelperTest(unittest.TestCase):
    def test_every_image_is_used_once_per_epoch_and_the_plan_is_seeded(self) -> None:
        first = np.concatenate([batch_positions(10, 5, i, seed=0) for i in range(2)])
        second = np.concatenate([batch_positions(10, 5, i, seed=0) for i in range(2, 4)])
        self.assertEqual(sorted(first), list(range(10)))
        self.assertEqual(sorted(second), list(range(10)))
        self.assertTrue(np.array_equal(batch_positions(10, 4, 3, seed=1), batch_positions(10, 4, 3, seed=1)))
        self.assertFalse(np.array_equal(batch_positions(10, 4, 3, seed=1), batch_positions(10, 4, 3, seed=2)))

    def test_ema_copies_the_generator_at_first_and_then_averages_over_ema_kimg(self) -> None:
        self.assertEqual(ema_beta(64, nimg=0, ema_kimg=500, rampup=0.05), 0.0)
        self.assertAlmostEqual(ema_beta(64, nimg=10**9, ema_kimg=500, rampup=0.05), 0.5 ** (64 / 500_000))
        self.assertAlmostEqual(ema_beta(64, nimg=1000, ema_kimg=500, rampup=0.05), 0.5 ** (64 / 50))

    def test_lazy_regularization_rescales_lr_and_betas(self) -> None:
        lr, betas = lazy_regularization(0.0025, (0.0, 0.99), 16)
        self.assertAlmostEqual(lr, 0.0025 * 16 / 17)
        self.assertEqual(betas[0], 0.0)
        self.assertAlmostEqual(betas[1], 0.99 ** (16 / 17))

    def test_settings_follow_the_cifar_configuration(self) -> None:
        s = make_settings(budget="50", seed=0, kimg=100, snapshot_kimg=10)
        self.assertEqual((s["batch_size"], s["mbstd_group"], s["lr"], s["r1_gamma"]), (64, 32, 0.0025, 0.01))
        self.assertEqual((s["ema_kimg"], s["ema_rampup"], s["mapping_layers"], s["ada_target"]), (500, 0.05, 2, 0.6))


class RunTrainingTest(unittest.TestCase):
    def test_a_run_writes_history_best_generator_and_result(self) -> None:
        reals, labels = tiny_data()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_training(tiny_settings(), {}, reals, labels, run_dir, torch.device("cpu"), mean_pixel_fid)
            history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
            self.assertEqual([h["iteration"] for h in history], [2, 4, 6])
            self.assertEqual(result["best_fid"], min(h["fid"] for h in history))
            self.assertTrue(result["reload_check_passed"])
            for name in ("config.json", "train.log", "best.pt", "result.json"):
                self.assertTrue((run_dir / name).exists(), name)
            self.assertFalse((run_dir / "last.pt").exists())
            with self.assertRaises(FileExistsError):
                run_training(tiny_settings(), {}, reals, labels, run_dir, torch.device("cpu"), mean_pixel_fid)

    def test_an_interrupted_run_resumes_to_the_same_generator(self) -> None:
        reals, labels = tiny_data()
        calls = []

        def crash_on_second_snapshot(images: np.ndarray) -> float:
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return mean_pixel_fid(images)

        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(0)
            run_training(tiny_settings(), {}, reals, labels, Path(tmp) / "a", torch.device("cpu"), mean_pixel_fid)
            torch.manual_seed(0)
            with self.assertRaises(KeyboardInterrupt):
                run_training(tiny_settings(), {}, reals, labels, Path(tmp) / "b", torch.device("cpu"),
                             crash_on_second_snapshot)
            self.assertTrue((Path(tmp) / "b" / "last.pt").exists())
            torch.manual_seed(123)  # the resumed run must not depend on the global state at restart
            run_training(tiny_settings(), {}, reals, labels, Path(tmp) / "b", torch.device("cpu"), mean_pixel_fid)
            a = torch.load(Path(tmp) / "a" / "best.pt", weights_only=True)
            b = torch.load(Path(tmp) / "b" / "best.pt", weights_only=True)
            self.assertEqual(a["kimg"], b["kimg"])
            for key in a["G_ema"]:
                torch.testing.assert_close(a["G_ema"][key], b["G_ema"][key], rtol=0, atol=0)
            history = (Path(tmp) / "b" / "history.jsonl").read_text().splitlines()
            self.assertEqual(len(history), 3)

    def test_changed_settings_are_refused(self) -> None:
        reals, labels = tiny_data()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(json.dumps({"settings": tiny_settings(seed=1)}))
            with self.assertRaises(ValueError):
                run_training(tiny_settings(), {}, reals, labels, run_dir, torch.device("cpu"), mean_pixel_fid)


if __name__ == "__main__":
    unittest.main()
