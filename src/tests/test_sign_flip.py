"""Tests for the sign-flip diagnostics (no GPU, dataset or checkpoints needed)."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.component.sign_flip import (
    add_randaugment_changes,
    class_quality,
    correlation_rows,
    correlation_table,
    holm,
    permutation_p,
    plot_recall_vs_change,
    rank,
    read_changes,
    reference_indices,
    spearman,
    synthetic_accuracy_measures,
)


def blobs(rng: np.random.Generator, classes: int, per_class: int, spread: list[float], shift: list[float],
          dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian points per class: centre 10 * class (+ shift), standard deviation spread[class]."""
    points, labels = [], []
    for c in range(classes):
        centre = np.full(dim, 10.0 * c + shift[c])
        points.append(centre + spread[c] * rng.standard_normal((per_class, dim)))
        labels += [c] * per_class
    return np.concatenate(points), np.array(labels)


class RankTest(unittest.TestCase):
    def test_ties_get_their_average_rank(self) -> None:
        np.testing.assert_allclose(rank(np.array([3.0, 1.0, 3.0, 2.0])), [3.5, 1.0, 3.5, 2.0])


class SpearmanTest(unittest.TestCase):
    def test_monotone_relations_give_plus_and_minus_one(self) -> None:
        x = np.arange(10.0)
        self.assertAlmostEqual(spearman(x, x ** 3), 1.0)
        self.assertAlmostEqual(spearman(x, -np.exp(x)), -1.0)

    def test_matches_the_textbook_formula_without_ties(self) -> None:
        x, y = np.array([1.0, 2, 3, 4, 5]), np.array([2.0, 1, 4, 3, 5])
        d = rank(x) - rank(y)
        self.assertAlmostEqual(spearman(x, y), 1 - 6 * (d ** 2).sum() / (5 * (25 - 1)))

    def test_constant_input_gives_nan(self) -> None:
        self.assertTrue(np.isnan(spearman(np.ones(5), np.arange(5.0))))
        self.assertTrue(np.isnan(permutation_p(np.ones(5), np.arange(5.0), permutations=10)))


class PermutationTest(unittest.TestCase):
    def test_strong_relation_is_significant_and_noise_is_not(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(100)
        self.assertLess(permutation_p(x, x + 0.1 * rng.standard_normal(100), permutations=2000), 0.001)
        self.assertGreater(permutation_p(x, rng.standard_normal(100), permutations=2000), 0.05)

    def test_p_value_is_never_zero(self) -> None:
        x = np.arange(20.0)
        self.assertAlmostEqual(permutation_p(x, x, permutations=999), 1 / 1000)


class ClassQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.real, self.real_labels = blobs(rng, 3, 60, [1.0, 1.0, 1.0], [0, 0, 0])
        # class 0: like the real class; class 1: collapsed to a small region; class 2: moved away
        self.fake, self.fake_labels = blobs(rng, 3, 60, [1.0, 0.1, 1.0], [0, 0, 4.0])
        self.rows = class_quality(self.real, self.fake, self.real_labels, self.fake_labels)

    def test_one_row_per_class_with_counts(self) -> None:
        self.assertEqual([r["class"] for r in self.rows], [0, 1, 2])
        self.assertEqual({(r["real_images"], r["generated_images"]) for r in self.rows}, {(60, 60)})

    def test_collapsed_class_has_low_recall_and_diversity(self) -> None:
        good, collapsed = self.rows[0], self.rows[1]
        self.assertGreater(good["recall"], 0.5)
        self.assertLess(collapsed["recall"], 0.2)
        self.assertLess(collapsed["diversity_ratio"], 0.2)
        self.assertAlmostEqual(good["diversity_ratio"], 1.0, delta=0.15)

    def test_moved_class_has_low_precision_and_a_large_centroid_gap(self) -> None:
        good, moved = self.rows[0], self.rows[2]
        self.assertLess(moved["precision"], good["precision"])
        self.assertGreater(moved["centroid_gap"], 3 * good["centroid_gap"])

    def test_too_few_images_raises(self) -> None:
        with self.assertRaises(ValueError):
            class_quality(self.real[:3], self.fake[:3], self.real_labels[:3], self.fake_labels[:3])


class ReferenceIndicesTest(unittest.TestCase):
    def test_draws_the_same_count_per_class_from_the_pool_only(self) -> None:
        labels = np.repeat(np.arange(4), 10)
        pool = np.flatnonzero(np.tile(np.arange(10) >= 2, 4))  # first two of each class are "validation"
        chosen = reference_indices(labels, pool, 5, num_classes=4)
        np.testing.assert_array_equal(np.bincount(labels[chosen]), [5, 5, 5, 5])
        self.assertTrue(set(chosen) <= set(pool))
        np.testing.assert_array_equal(chosen, reference_indices(labels, pool, 5, num_classes=4))

    def test_short_class_raises(self) -> None:
        with self.assertRaises(ValueError):
            reference_indices(np.repeat(np.arange(2), 3), np.arange(6), 4, num_classes=2)


def quality_rows(recall: np.ndarray) -> list[dict]:
    return [{"class": str(c), "precision": str(0.5 + 0.001 * c), "recall": str(v), "diversity_ratio": str(v),
             "centroid_gap": str(1 - v)} for c, v in enumerate(recall)]


class CorrelationTest(unittest.TestCase):
    def test_finds_the_relation_in_scratch_and_not_in_pretrained(self) -> None:
        rng = np.random.default_rng(0)
        recall = rng.uniform(0, 1, 100)
        changes = {("scratch", "sd_prompt", "randaugment"): {"resnet50": 20 * recall + rng.standard_normal(100)},
                   ("pretrained", "sd_prompt", "real_only"): {"resnet50": rng.standard_normal(100)}}
        synthetic = {("scratch", "resnet50"): recall * 50}
        rows = correlation_rows(quality_rows(recall), changes, synthetic, permutations=500)
        cell = {(r["init"], r["measure"]): r for r in rows}
        self.assertGreater(cell[("scratch", "recall")]["spearman"], 0.9)
        self.assertLess(cell[("scratch", "recall")]["p_value"], 0.01)
        self.assertLess(cell[("scratch", "centroid_gap")]["spearman"], -0.9)
        self.assertIn(("scratch", "synthetic_accuracy"), cell)
        self.assertNotIn(("pretrained", "synthetic_accuracy"), cell)
        self.assertLess(abs(cell[("pretrained", "recall")]["spearman"]), 0.3)
        self.assertEqual(cell[("pretrained", "recall")]["classes"], 100)

    def test_classes_missing_from_the_change_are_left_out(self) -> None:
        change = np.arange(100.0)
        change[:10] = np.nan
        rows = correlation_rows(quality_rows(np.linspace(0, 1, 100)),
                                {("scratch", "sd_prompt", "real_only"): {"vit_b_16": change}}, permutations=100)
        self.assertEqual({r["classes"] for r in rows}, {90})

    def test_table_has_one_row_per_measure(self) -> None:
        rows = correlation_rows(quality_rows(np.linspace(0, 1, 100)),
                                {("scratch", "sd_prompt", "real_only"): {"vit_b_16": np.arange(100.0)}},
                                permutations=100)
        table = correlation_table(rows)
        self.assertIn(r"\label{tab:sign-flip}", table)
        self.assertEqual(sum(line.startswith(("recall", "precision", "diversity", "centroid")) for line in
                             table.splitlines()), 4)


class HolmTest(unittest.TestCase):
    def test_matches_the_step_down_procedure(self) -> None:
        # sorted 0.01, 0.02, 0.04 -> 3*0.01, max(0.03, 2*0.02), max(0.04, 1*0.04)
        np.testing.assert_allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])

    def test_caps_at_one_and_skips_nan(self) -> None:
        adjusted = holm([0.5, float("nan"), 0.6])
        self.assertTrue(np.isnan(adjusted[1]))
        np.testing.assert_allclose([adjusted[0], adjusted[2]], [1.0, 1.0])

    def test_rows_get_an_adjusted_p_value(self) -> None:
        rows = correlation_rows(quality_rows(np.linspace(0, 1, 100)),
                                {("scratch", "sd_prompt", "real_only"): {"vit_b_16": np.arange(100.0)}},
                                permutations=100)
        self.assertTrue(all(r["p_holm"] >= r["p_value"] for r in rows))
        self.assertIn("4 tests", correlation_table(rows))


class RandAugmentChangeTest(unittest.TestCase):
    def test_derives_randaugment_minus_real_from_both_sd_tables(self) -> None:
        changes = {("scratch", "sd_prompt", "real_only"): {"resnet50": np.array([5.0, -2.0])},
                   ("scratch", "sd_prompt", "randaugment"): {"resnet50": np.array([1.0, -6.0])},
                   ("pretrained", "sd_prompt", "real_only"): {"resnet50": np.array([3.0, 3.0])}}
        add_randaugment_changes(changes)
        np.testing.assert_allclose(changes[("scratch", "randaugment", "real_only")]["resnet50"], [4.0, 4.0])
        self.assertNotIn(("pretrained", "randaugment", "real_only"), changes)  # its SD - RA table is missing


class FilesTest(unittest.TestCase):
    def test_read_changes_indexes_by_class_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classes.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["model", "class", "class_name", "superclass_name",
                                                       "recall_change", "seeds"])
                writer.writeheader()
                writer.writerow({"model": "resnet50", "class": 7, "class_name": "x", "superclass_name": "y",
                                 "recall_change": -12.5, "seeds": "0 1 2"})
            changes = read_changes(path)
        self.assertEqual(changes["resnet50"][7], -12.5)
        self.assertTrue(np.isnan(changes["resnet50"][0]))

    def test_synthetic_accuracy_uses_one_budget_and_averages_its_runs(self) -> None:
        rows = [{"init": "scratch", "model": "resnet50", "budget": "full", "synthetic_class_accuracy": "10 20"},
                {"init": "scratch", "model": "resnet50", "budget": "full", "synthetic_class_accuracy": "30 40"},
                {"init": "scratch", "model": "resnet50", "budget": "5", "synthetic_class_accuracy": "0 0"}]
        np.testing.assert_allclose(synthetic_accuracy_measures(rows)[("scratch", "resnet50")], [20, 30])
        np.testing.assert_allclose(synthetic_accuracy_measures(rows, "5")[("scratch", "resnet50")], [0, 0])

    def test_plot_writes_svg_and_pdf(self) -> None:
        changes = {("scratch", "sd_prompt", "real_only"): {"resnet50": np.arange(100.0)},
                   ("pretrained", "sd_prompt", "real_only"): {"resnet50": -np.arange(100.0)}}
        with tempfile.TemporaryDirectory() as tmp:
            paths = plot_recall_vs_change(quality_rows(np.linspace(0, 1, 100)), changes,
                                          ("sd_prompt", "real_only"), Path(tmp) / "fig")
            self.assertEqual([p.suffix for p in paths], [".svg", ".pdf"])
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))


if __name__ == "__main__":
    unittest.main()
