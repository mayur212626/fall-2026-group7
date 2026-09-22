import json
import unittest
from pathlib import Path

import numpy as np

from scripts.metrics import classification_metrics
from scripts.prepare_cifar100 import make_splits


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.tile(np.arange(100), 500)
        config_path = Path(__file__).resolve().parents[1] / "configs/cifar100_fewshot.json"
        self.config = json.loads(config_path.read_text())

    def test_split_counts_disjointness_and_nesting(self):
        result = make_splits(self.targets, self.config)
        validation = result["validation_indices"]
        self.assertEqual(len(validation), len(set(validation)))
        np.testing.assert_array_equal(np.bincount(self.targets[validation]), np.full(100, 50))
        previous = set()
        for shots in (5, 10, 20, 50):
            indices = result["train_indices"][str(shots)]
            selected = set(indices)
            self.assertEqual(len(indices), 100 * shots)
            self.assertEqual(len(indices), len(selected))
            self.assertTrue(previous <= selected)
            self.assertFalse(selected.intersection(validation))
            self.assertGreaterEqual(min(indices), 0)
            self.assertLess(max(indices), len(self.targets))
            np.testing.assert_array_equal(np.bincount(self.targets[indices]), np.full(100, shots))
            previous = selected

    def test_seed_reproducibility(self):
        first = make_splits(self.targets, self.config)
        self.assertEqual(first, make_splits(self.targets, self.config))
        different = make_splits(self.targets, {**self.config, "split_seed": 43})
        self.assertNotEqual(first["validation_indices"], different["validation_indices"])

    def test_full_training_pool_keeps_validation_and_fewshot_splits(self):
        original = make_splits(self.targets, self.config)
        result = make_splits(self.targets, self.config, include_full=True)
        full_indices = result.pop("full_train_indices")
        self.assertEqual(result, original)
        self.assertEqual(len(full_indices), 45000)
        self.assertEqual(len(set(full_indices)), 45000)
        np.testing.assert_array_equal(np.bincount(self.targets[full_indices]), np.full(100, 450))
        full_training = set(full_indices)
        validation = set(result["validation_indices"])
        self.assertFalse(full_training.intersection(validation))
        self.assertEqual(full_training | validation, set(range(50000)))
        self.assertTrue(set(result["train_indices"]["50"]) <= full_training)

    def test_full_training_pool_follows_the_validation_budget(self):
        config = {**self.config, "validation_per_class": 25}
        result = make_splits(self.targets, config, include_full=True)
        indices = result["full_train_indices"]
        self.assertEqual(len(indices), 47500)
        self.assertFalse(set(indices).intersection(result["validation_indices"]))
        np.testing.assert_array_equal(np.bincount(self.targets[indices]), np.full(100, 475))

    def test_invalid_data_and_budgets(self):
        with self.assertRaises(ValueError):
            make_splits(self.targets[:-1], self.config)
        with self.assertRaises(ValueError):
            make_splits(np.zeros(50000, dtype=int), self.config)
        with self.assertRaises(ValueError):
            make_splits(self.targets, {**self.config, "validation_per_class": 496})
        with self.assertRaises(ValueError):
            make_splits(self.targets, {**self.config, "shots_per_class": [5, 5]})


class MetricTests(unittest.TestCase):
    def test_hand_calculated_imbalanced_case(self):
        result = classification_metrics([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 2], 3)
        self.assertAlmostEqual(result["accuracy"], 100 * 4 / 6)
        self.assertAlmostEqual(result["balanced_accuracy"], 100 * (2 / 3 + 1 / 2 + 1) / 3)
        self.assertAlmostEqual(result["macro_f1"], 100 * (4 / 5 + 1 / 2 + 2 / 3) / 3)
        self.assertEqual(result["class_counts"], [3, 2, 1])
        self.assertEqual(result["confusion_matrix"], [[2, 1, 0], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_allclose(result["per_class_recall"], [200 / 3, 50, 100])

    def test_perfect_predictions(self):
        result = classification_metrics([0, 1, 2], [0, 1, 2], 3)
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            self.assertEqual(result[metric], 100)

    def test_class_with_no_correct_predictions(self):
        result = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0], 2)
        self.assertEqual(result["per_class_recall"], [100, 0])
        self.assertEqual(result["accuracy"], result["balanced_accuracy"])
        self.assertAlmostEqual(result["macro_f1"], 100 / 3)

    def test_invalid_predictions_and_missing_class(self):
        cases = [([], [], 2), ([0], [0, 1], 2), ([0, 1], [0, 2], 2),
                 ([0, 1], [0, -1], 2), ([0, 1], [0.0, 1.0], 2), ([0], [0], 2)]
        for targets, predictions, classes in cases:
            with self.subTest(targets=targets, predictions=predictions):
                with self.assertRaises(ValueError):
                    classification_metrics(targets, predictions, classes)

    def test_small_integer_labels_do_not_overflow(self):
        targets = np.arange(100, dtype=np.uint8)
        result = classification_metrics(targets, targets, 100)
        self.assertEqual(result["accuracy"], 100)
        self.assertEqual(result["macro_f1"], 100)


if __name__ == "__main__":
    unittest.main()
