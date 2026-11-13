"""Tests for the ensembles of real-data and generated-data classifiers (numpy only, no checkpoints)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.component import ensemble as en

LABELS = np.arange(10) % 5  # 10 images, 5 classes


def probs(correct: list[int], wrong_class: int = 0) -> np.ndarray:
    """Probabilities that put 0.6 on the true class for the images in ``correct`` and 0.6 elsewhere otherwise."""
    p = np.full((10, 5), 0.1, dtype=np.float32)
    for i, label in enumerate(LABELS):
        p[i, label if i in correct else (label + 1 + wrong_class) % 5] = 0.6
    return p


def run(p: np.ndarray, ids: np.ndarray | None = None) -> dict:
    return {"probs": p, "ids": np.arange(10) if ids is None else ids, "labels": LABELS}


class EnsembleScoresTest(unittest.TestCase):
    def test_singles_and_average_of_two_members(self) -> None:
        runs = {("pretrained", "resnet50", "5", "real_only", 0): run(probs([0, 1, 2, 3, 4])),
                ("pretrained", "resnet50", "5", "sd_prompt", 0): run(probs([5, 6, 7, 8, 9]))}
        rows = {(r["condition"], r["seed"]): r for r in en.ensemble_scores(runs)}
        self.assertEqual(rows[("real_only", 0)]["accuracy"], 50.0)
        # each member is right on its half; on the other half its wrong 0.6 meets the other's 0.6 on the
        # true class and ties break to the lower class index, so check against the explicit average
        average = (probs([0, 1, 2, 3, 4]) + probs([5, 6, 7, 8, 9])) / 2
        self.assertAlmostEqual(rows[("real+SD", 0)]["accuracy"], en.accuracy(average, LABELS))
        self.assertNotIn(("real+real", 0), rows)  # one real-only seed: no control ensemble

    def test_identical_members_pair_each_seed_with_the_next(self) -> None:
        runs = {("scratch", "vit_b_16", "50", "real_only", s): run(probs([s])) for s in range(3)}
        rows = [r for r in en.ensemble_scores(runs) if r["condition"] == "real+real"]
        self.assertEqual(sorted((r["seed"], r["partner_seed"]) for r in rows), [(0, 1), (1, 2), (2, 0)])

    def test_members_on_different_images_are_refused(self) -> None:
        runs = {("pretrained", "resnet50", "5", "real_only", 0): run(probs([0])),
                ("pretrained", "resnet50", "5", "sd_prompt", 0): run(probs([0]), ids=np.arange(10)[::-1])}
        with self.assertRaises(ValueError):
            en.ensemble_scores(runs)


class ComparisonTest(unittest.TestCase):
    def test_real_plus_sd_against_real_plus_real_is_paired_by_seed(self) -> None:
        runs = {}
        for s in range(3):
            runs[("pretrained", "resnet50", "5", "real_only", s)] = run(probs([0, 1, 2, 3, 4], wrong_class=0))
            runs[("pretrained", "resnet50", "5", "sd_prompt", s)] = run(probs([5, 6, 7, 8, 9], wrong_class=1))
        changes = {c["label"]: c for c in en.comparisons(en.ensemble_scores(runs))}
        c = changes["(real+SD)$-$(real+real)"]
        self.assertEqual(c["seeds"], [0, 1, 2])
        self.assertGreater(c["mean"], 0)
        self.assertEqual(c["verdict"], "gain")


class CombineTest(unittest.TestCase):
    def test_reads_exported_files_and_writes_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "stage1-pretrained"
            for s in range(3):
                members = (("randaugment", [0, 1, 2, 3, 4], 0), ("sd_prompt", [5, 6, 7, 8, 9], 1))
                for condition, correct, wrong in members:
                    d = root / f"pretrained_resnet50_{condition}_b5_s{s}"
                    d.mkdir(parents=True)
                    settings = {"init": "pretrained", "model": "resnet50", "budget": "5", "condition": condition,
                                "seed": s}
                    (d / "config.json").write_text(json.dumps({"settings": settings}))
                    np.savez(d / "probs.npz", validation_probs=probs(correct, wrong).astype(np.float16),
                             validation_ids=np.arange(10), validation_labels=LABELS)
            tables = Path(tmp) / "tables"
            tables.mkdir()
            changes = en.combine([f"{tmp}/stage1-{{init}}"], "validation", tables)
            self.assertEqual({c["label"] for c in changes}, {"(RA+SD)$-$(RA+RA)"})
            text = (tables / "ensemble_pretrained.tex").read_text()
            self.assertIn(r"\label{tab:ensemble-pretrained}", text)
            self.assertTrue((tables / "ensemble_accuracy.csv").exists())
            with self.assertRaises(SystemExit):
                en.combine([f"{tmp}/stage1-{{init}}"], "test", tables)  # nothing exported for test


if __name__ == "__main__":
    unittest.main()
