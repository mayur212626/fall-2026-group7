"""Tests for the classification metrics."""

import unittest

from src.component.metrics import classification_metrics


class HandCalculatedExampleTest(unittest.TestCase):
    """Three classes, six images, checked by hand.

    Confusion matrix (rows true, columns predicted):
        class 0: [1, 1, 0]   recall 1/2, precision 1/2, F1 1/2
        class 1: [0, 2, 0]   recall 2/2, precision 2/3, F1 4/5
        class 2: [1, 0, 1]   recall 1/2, precision 1/1, F1 2/3
    """

    def setUp(self) -> None:
        self.metrics = classification_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2], 3)

    def test_confusion_matrix(self) -> None:
        self.assertEqual(self.metrics["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])

    def test_accuracy(self) -> None:
        self.assertAlmostEqual(self.metrics["accuracy"], 100 * 4 / 6)

    def test_per_class_recall(self) -> None:
        self.assertEqual(self.metrics["per_class_recall"], [50.0, 100.0, 50.0])

    def test_balanced_accuracy(self) -> None:
        self.assertAlmostEqual(self.metrics["balanced_accuracy"], 100 * 2 / 3)

    def test_macro_f1_is_mean_of_class_f1(self) -> None:
        self.assertAlmostEqual(self.metrics["macro_f1"], 100 * (1 / 2 + 4 / 5 + 2 / 3) / 3)

    def test_class_counts(self) -> None:
        self.assertEqual(self.metrics["class_counts"], [2, 2, 2])


class PerfectPredictionsTest(unittest.TestCase):
    def test_all_scores_are_100(self) -> None:
        metrics = classification_metrics([0, 1, 2, 2], [0, 1, 2, 2], 3)
        for name in ("accuracy", "balanced_accuracy", "macro_f1"):
            self.assertEqual(metrics[name], 100.0, name)
        self.assertEqual(metrics["per_class_recall"], [100.0, 100.0, 100.0])


class ClassWithNoCorrectPredictionTest(unittest.TestCase):
    """Class 1 is never predicted: recall 0, precision undefined, F1 counted as 0."""

    def setUp(self) -> None:
        self.metrics = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0], 2)

    def test_recall_is_zero(self) -> None:
        self.assertEqual(self.metrics["per_class_recall"], [100.0, 0.0])

    def test_undefined_f1_counts_as_zero(self) -> None:
        # Class 0: precision 1/2, recall 1, F1 2/3. Class 1: F1 0.
        self.assertAlmostEqual(self.metrics["macro_f1"], 100 * (2 / 3 + 0) / 2)

    def test_balanced_accuracy(self) -> None:
        self.assertEqual(self.metrics["balanced_accuracy"], 50.0)


class MissingEvaluationClassTest(unittest.TestCase):
    """Class 2 has no evaluation images."""

    def setUp(self) -> None:
        self.metrics = classification_metrics([0, 1], [0, 1], 3)

    def test_missing_class_is_reported(self) -> None:
        self.assertEqual(self.metrics["missing_classes"], [2])

    def test_missing_class_has_no_recall(self) -> None:
        self.assertEqual(self.metrics["per_class_recall"], [100.0, 100.0, None])

    def test_balanced_accuracy_uses_present_classes(self) -> None:
        self.assertEqual(self.metrics["balanced_accuracy"], 100.0)

    def test_macro_f1_keeps_full_class_list(self) -> None:
        self.assertAlmostEqual(self.metrics["macro_f1"], 100 * 2 / 3)


class InputValidationTest(unittest.TestCase):
    def test_rejects_label_outside_class_range(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics([0, 3], [0, 0], 3)

    def test_rejects_different_lengths(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0], 3)

    def test_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics([], [], 3)


if __name__ == "__main__":
    unittest.main()
