"""Tests for the Stage 3 run list and its split across instances (standard library only)."""

import unittest

from src.component import stage3_jobs as sj


class JobsTest(unittest.TestCase):
    def test_grid_size(self) -> None:
        seeds = [0, 1, 2, 3, 4]
        self.assertEqual(len(sj.all_jobs(sj.load_steps("flowers102"), seeds)), 2 * 2 * 3 * 10 * 5)
        self.assertEqual(len(sj.all_jobs(sj.load_steps("ham10000"), seeds)), 2 * 2 * 4 * 10 * 5)

    def test_names_follow_the_training_runner(self) -> None:
        names = {j["name"] for j in sj.all_jobs(sj.load_steps("flowers102"), [0])}
        self.assertIn("pretrained_resnet50_real_only_b2_s0", names)
        self.assertIn("scratch_vit_b_16_sd_prompt_randaugment_b10_s0_r5", names)
        self.assertIn("scratch_vit_b_16_sd_prompt_randaugment_b10_s0", names)
        self.assertEqual(len(names), 120)

    def test_cost_is_steps_times_seconds_per_step(self) -> None:
        jobs = sj.all_jobs({"pretrained": {"5": 3600}, "scratch": {"5": 7200}}, [0])
        hours = {(j["init"], j["model"]): j["hours"] for j in jobs}
        self.assertAlmostEqual(hours[("pretrained", "vit_b_16")], sj.SECONDS_PER_STEP["vit_b_16"])
        self.assertAlmostEqual(hours[("scratch", "resnet50")], 2 * sj.SECONDS_PER_STEP["resnet50"])


class SplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = sj.all_jobs(sj.load_steps("ham10000"), [0, 1, 2])

    def test_parts_are_disjoint_and_complete(self) -> None:
        parts = sj.split(self.jobs, 3)
        names = [j["name"] for part in parts for j in part]
        self.assertEqual(sorted(names), sorted(j["name"] for j in self.jobs))

    def test_parts_are_balanced_after_every_seed(self) -> None:
        parts = sj.split(self.jobs, 3)
        for seed in (0, 1, 2):
            hours = [sum(j["hours"] for j in part if j["seed"] <= seed) for part in parts]
            self.assertLess(max(hours) - min(hours), 0.05 * max(hours))

    def test_parts_keep_the_grid_order(self) -> None:
        position = {j["name"]: i for i, j in enumerate(self.jobs)}
        for part in sj.split(self.jobs, 2):
            order = [position[j["name"]] for j in part]
            self.assertEqual(order, sorted(order))

    def test_one_part_is_the_whole_grid(self) -> None:
        self.assertEqual(sj.split(self.jobs, 1), [self.jobs])

    def test_same_arguments_same_split(self) -> None:
        self.assertEqual(sj.split(self.jobs, 3), sj.split(sj.all_jobs(sj.load_steps("ham10000"), [0, 1, 2]), 3))


if __name__ == "__main__":
    unittest.main()
