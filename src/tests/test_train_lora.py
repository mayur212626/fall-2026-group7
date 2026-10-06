"""Tests for the LoRA training plan and captions (NumPy only; no model is loaded)."""

import unittest

from src.component.generate_sd import TEMPLATES
from src.component.train_lora import STEPS, captions, training_plan


class TrainingPlanTest(unittest.TestCase):
    def test_every_image_is_used_once_per_epoch(self) -> None:
        plan = training_plan(num_items=10, batch_size=5, steps=4, seed=0)
        positions = [p for step in plan for p in step["positions"]]
        self.assertEqual(sorted(positions[:10]), list(range(10)))
        self.assertEqual(sorted(positions[10:]), list(range(10)))

    def test_same_seed_same_plan_and_different_seed_differs(self) -> None:
        self.assertEqual(training_plan(20, 4, 6, seed=1), training_plan(20, 4, 6, seed=1))
        self.assertNotEqual(training_plan(20, 4, 6, seed=1), training_plan(20, 4, 6, seed=2))

    def test_templates_and_flips_are_valid(self) -> None:
        plan = training_plan(50, 8, 20, seed=0)
        templates = {t for step in plan for t in step["templates"]}
        flips = {f for step in plan for f in step["flips"]}
        self.assertTrue(templates <= set(range(len(TEMPLATES))) and len(templates) > 1)
        self.assertEqual(flips, {True, False})

    def test_every_budget_has_a_step_count(self) -> None:
        self.assertEqual(sorted(STEPS), sorted(["5", "10", "20", "50", "full"]))


class CaptionsTest(unittest.TestCase):
    def test_identifier_goes_before_the_class_name(self) -> None:
        names = ["apple", "aquarium_fish", "bear"]
        self.assertEqual(captions(names, [2, 0], [0, 0], "cfr"),
                         ["a photo of a cfr bear.", "a photo of a cfr apple."])


if __name__ == "__main__":
    unittest.main()
