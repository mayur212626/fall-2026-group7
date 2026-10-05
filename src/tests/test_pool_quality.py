"""Tests for the pool-level quality helpers: matched references, bootstrap FID SD and the table.

FID and precision/recall themselves are tested in test_image_quality.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.component.image_quality import precision_recall
from src.component.pool_quality import append_table, bootstrap_fid_sd, matched_reference

RNG = np.random.default_rng(0)
A = RNG.normal(size=(400, 8))


class ChunkedPrecisionRecallTest(unittest.TestCase):
    def test_chunking_does_not_change_the_scores(self) -> None:
        from src.component import image_quality

        fake = RNG.normal(size=(300, 8)) + 0.3
        whole = (image_quality._inside(fake, A, 3, chunk=10_000), image_quality._inside(A, fake, 3, chunk=10_000))
        chunked = (image_quality._inside(fake, A, 3, chunk=7), image_quality._inside(A, fake, 3, chunk=7))
        self.assertEqual(whole, chunked)
        self.assertEqual(precision_recall(A, fake), chunked)

    def test_collapsed_generator_has_high_precision_but_low_recall(self) -> None:
        collapsed = np.repeat(A[:5], 80, axis=0) + 1e-3 * RNG.normal(size=(400, 8))
        precision, recall = precision_recall(A, collapsed)
        self.assertGreater(precision, 0.9)
        self.assertLess(recall, 0.2)


class BootstrapTest(unittest.TestCase):
    def test_sd_is_positive_and_skipped_below_two_resamples(self) -> None:
        self.assertGreater(bootstrap_fid_sd(A, A + 0.5, 5), 0)
        self.assertIsNone(bootstrap_fid_sd(A, A, 1))


class MatchedReferenceTest(unittest.TestCase):
    labels = np.repeat(np.arange(4), 30)

    def test_class_balanced_disjoint_samples(self) -> None:
        refs = matched_reference(self.labels, np.arange(120), per_class=10, num_classes=4, seed=0)
        self.assertEqual(len(refs), 2)
        for ref in refs:
            self.assertEqual(np.bincount(self.labels[ref]).tolist(), [10, 10, 10, 10])
        self.assertEqual(len(np.intersect1d(refs[0], refs[1])), 0)

    def test_one_sample_when_the_pool_is_too_small_for_two(self) -> None:
        self.assertEqual(len(matched_reference(self.labels, np.arange(120), 20, 4, 0)), 1)

    def test_only_pool_indices_are_used(self) -> None:
        pool = np.arange(0, 120, 2)
        ref = matched_reference(self.labels, pool, 5, 4, 0)[0]
        self.assertTrue(np.isin(ref, pool).all())


class TableTest(unittest.TestCase):
    def test_same_name_replaces_the_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.csv"
            append_table(path, {"name": "a", "fid": 10})
            append_table(path, {"name": "b", "fid": 20})
            append_table(path, {"name": "a", "fid": 5})
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([(r["name"], r["fid"]) for r in rows], [("b", "20"), ("a", "5")])


if __name__ == "__main__":
    unittest.main()
