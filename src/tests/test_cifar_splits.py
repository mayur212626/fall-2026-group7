"""Tests for the CIFAR-100 split manifest builder."""

import unittest

import numpy as np

from src.component.cifar_splits import build_split_manifest, labels_sha256

NUM_CLASSES = 3
IMAGES_PER_CLASS = 100
VALIDATION_PER_CLASS = 10
SHOTS = [2, 5, 20]


def make_labels(seed: int = 0) -> list[int]:
    """Return shuffled labels with IMAGES_PER_CLASS images for each class."""
    labels = np.repeat(np.arange(NUM_CLASSES), IMAGES_PER_CLASS)
    return np.random.default_rng(seed).permutation(labels).tolist()


def build(labels: list[int], seed: int = 42) -> dict:
    """Build a manifest with the test settings."""
    return build_split_manifest(labels, NUM_CLASSES, VALIDATION_PER_CLASS, SHOTS, seed)


def class_counts(labels: list[int], indices: list[int]) -> list[int]:
    """Count how many of the given indices belong to each class."""
    return np.bincount(np.asarray(labels)[indices], minlength=NUM_CLASSES).tolist()


class BuildSplitManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = make_labels()
        self.manifest = build(self.labels)

    def test_validation_has_requested_count_per_class(self) -> None:
        counts = class_counts(self.labels, self.manifest["validation_indices"])
        self.assertEqual(counts, [VALIDATION_PER_CLASS] * NUM_CLASSES)

    def test_each_budget_has_requested_count_per_class(self) -> None:
        for shots in SHOTS:
            counts = class_counts(self.labels, self.manifest["train_indices"][str(shots)])
            self.assertEqual(counts, [shots] * NUM_CLASSES, f"{shots} per class")

    def test_full_budget_is_everything_except_validation(self) -> None:
        full = set(self.manifest["train_indices"]["full"])
        validation = set(self.manifest["validation_indices"])
        self.assertEqual(full | validation, set(range(len(self.labels))))
        self.assertFalse(full & validation)

    def test_budgets_are_nested(self) -> None:
        budgets = [str(shots) for shots in SHOTS] + ["full"]
        for smaller, larger in zip(budgets, budgets[1:]):
            small = set(self.manifest["train_indices"][smaller])
            large = set(self.manifest["train_indices"][larger])
            self.assertTrue(small <= large, f"{smaller} not inside {larger}")

    def test_indices_are_sorted_without_duplicates(self) -> None:
        lists = [self.manifest["validation_indices"], *self.manifest["train_indices"].values()]
        for indices in lists:
            self.assertEqual(indices, sorted(set(indices)))

    def test_same_seed_gives_same_manifest(self) -> None:
        self.assertEqual(build(self.labels), self.manifest)

    def test_different_seed_gives_different_validation(self) -> None:
        other = build(self.labels, seed=7)
        self.assertNotEqual(other["validation_indices"], self.manifest["validation_indices"])

    def test_records_settings_and_label_hash(self) -> None:
        self.assertEqual(self.manifest["split_seed"], 42)
        self.assertEqual(self.manifest["validation_per_class"], VALIDATION_PER_CLASS)
        self.assertEqual(self.manifest["shots_per_class"], SHOTS)
        self.assertEqual(self.manifest["labels_sha256"], labels_sha256(self.labels))

    def test_rejects_class_with_too_few_images(self) -> None:
        labels = self.labels + [NUM_CLASSES]  # class 3 has a single image
        with self.assertRaises(ValueError):
            build_split_manifest(labels, NUM_CLASSES + 1, VALIDATION_PER_CLASS, SHOTS, 42)


class LabelsSha256Test(unittest.TestCase):
    def test_changes_when_one_label_changes(self) -> None:
        labels = make_labels()
        changed = list(labels)
        changed[0] = (changed[0] + 1) % NUM_CLASSES
        self.assertNotEqual(labels_sha256(labels), labels_sha256(changed))


if __name__ == "__main__":
    unittest.main()
