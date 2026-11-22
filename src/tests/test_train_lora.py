"""Tests for the LoRA training plan and captions (NumPy only; no model is loaded)."""

import unittest

from src.component.generate_sd import TEMPLATES
from src.component.train_lora import STAGE3_STEPS, STEPS, captions, lora_steps, training_plan


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


class Stage3LoraTest(unittest.TestCase):
    def test_steps_follow_the_cifar_budget(self) -> None:
        self.assertEqual(lora_steps("cifar100", "full"), STEPS["full"])
        self.assertEqual(lora_steps("ham10000", "50"), STEPS["50"])
        self.assertEqual(lora_steps("flowers102", "10"), STEPS["10"])
        self.assertEqual(lora_steps("flowers102", "2"), STEPS["5"])
        self.assertEqual(STAGE3_STEPS["2"], STEPS["5"])

    def test_budget_must_belong_to_the_dataset(self) -> None:
        with self.assertRaises(ValueError):
            lora_steps("flowers102", "50")
        with self.assertRaises(ValueError):
            lora_steps("ham10000", "2")

    def test_template_indices_cover_the_dataset_templates(self) -> None:
        plan = training_plan(30, 8, 50, seed=0, num_templates=3)
        used = {t for step in plan for t in step["templates"]}
        self.assertEqual(used, {0, 1, 2})

    def test_captions_use_the_dataset_templates(self) -> None:
        from src.component.stage3_data import DATASETS

        templates = DATASETS["ham10000"]["templates"]
        self.assertEqual(captions(["melanoma"], [0], [0], "cfr", templates), ["a dermoscopy image of cfr melanoma."])


if __name__ == "__main__":
    unittest.main()
