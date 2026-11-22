"""Tests for the pool memorization check (NumPy only; no model)."""

import unittest

import numpy as np

from src.component import pool_memorization as pm


class NearestTest(unittest.TestCase):
    def test_nearest_is_within_the_same_class(self) -> None:
        reference = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        query = np.array([[1.0, 0.1], [1.0, 0.0]])
        similarity, nearest = pm.nearest_similarity(query, [0, 1], reference, [0, 1, 1])
        self.assertEqual(nearest.tolist(), [0, 2])  # class 1 cannot match reference 0
        self.assertAlmostEqual(similarity[1], 1 / np.sqrt(2))

    def test_scale_does_not_matter(self) -> None:
        rng = np.random.default_rng(0)
        q, r = rng.normal(size=(5, 4)), rng.normal(size=(6, 4))
        a, _ = pm.nearest_similarity(q, [0] * 5, r, [0] * 6)
        b, _ = pm.nearest_similarity(10 * q, [0] * 5, 0.1 * r, [0] * 6)
        np.testing.assert_allclose(a, b)

    def test_missing_reference_class_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            pm.nearest_similarity(np.ones((1, 2)), [3], np.ones((1, 2)), [0])


class SummaryTest(unittest.TestCase):
    def test_copies_are_flagged(self) -> None:
        held_out = np.linspace(0.5, 0.8, 101)
        copied = pm.summarize(np.full(20, 0.99), held_out)
        self.assertEqual(copied["share_above_held_out_p99"], 1.0)
        self.assertEqual(copied["share_near_copy"], 1.0)
        self.assertTrue(pm.verdict(copied).startswith("WARNING"))

    def test_new_images_pass(self) -> None:
        held_out = np.linspace(0.5, 0.8, 101)
        fine = pm.summarize(np.linspace(0.45, 0.75, 50), held_out)
        self.assertEqual(fine["share_above_held_out_p99"], 0.0)
        self.assertEqual(pm.verdict(fine), "no sign of copying")
        self.assertAlmostEqual(fine["held_out_median"], 0.65)


if __name__ == "__main__":
    unittest.main()
