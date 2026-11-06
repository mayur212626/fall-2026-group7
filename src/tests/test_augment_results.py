"""Tests for the stacked-augmentation results (rows of scores, no runs or torch)."""

import unittest

from src.component import augment_results as au


def rows(init: str, budget: str, scores: dict[str, list[float]], model: str = "resnet50") -> list[dict]:
    return [{"init": init, "model": model, "budget": budget, "condition": c, "seed": s, "accuracy": v}
            for c, values in scores.items() for s, v in enumerate(values)]


class StudyTest(unittest.TestCase):
    def test_stacked_compares_each_policy_with_and_without_sd(self) -> None:
        data = rows("scratch", "5", {"real_only": [10, 11, 12], "trivialaugment": [15, 16, 17],
                                     "sd_prompt_trivialaugment": [17, 18, 19], "randaugment": [14, 15, 16]})
        changes = {c["label"]: c for c in au.study_changes(data, "stacked", "accuracy")}
        self.assertEqual(set(changes), {"SD+TA$-$TA"})
        self.assertAlmostEqual(changes["SD+TA$-$TA"]["mean"], 2.0)
        table = au.study_table(list(changes.values()), "stacked", "scratch", "accuracy", "validation")
        self.assertIn(r"\label{tab:augment-stacked-scratch}", table)
        self.assertIn(r"+2.00 $\pm$ 0.00 $\uparrow$", table)

    def test_cells_without_the_new_condition_are_left_out(self) -> None:
        data = rows("scratch", "20", {"real_only": [1, 2, 3], "randaugment": [2, 3, 4]})
        self.assertEqual(au.study_changes(data, "policies", "accuracy"), [])


class BestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = rows("pretrained", "5", {
            "real_only": [40, 41, 42], "randaugment": [44, 45, 46], "trivialaugment": [45, 46, 47],
            "sd_prompt": [48, 49, 50], "sd_prompt_trivialaugment": [50, 51, 52], "augmix": [60, 61]})  # 2 seeds

    def test_choices_use_validation_and_need_three_seeds(self) -> None:
        chosen = au.choose_best(self.validation, "accuracy")
        self.assertEqual(chosen[("pretrained", "resnet50", "5")],
                         {"conventional": "trivialaugment", "sd": "sd_prompt_trivialaugment"})

    def test_test_split_compares_the_validation_choices(self) -> None:
        chosen = au.choose_best(self.validation, "accuracy")
        test = rows("pretrained", "5", {"real_only": [39, 40, 41], "randaugment": [50, 51, 52],
                                        "trivialaugment": [43, 44, 45], "sd_prompt_trivialaugment": [47, 49, 48]})
        (c,) = au.best_changes(test, chosen, "accuracy")
        self.assertEqual((c["treatment"], c["control"], c["label"]),
                         ("sd_prompt_trivialaugment", "trivialaugment", "SD+TA$-$TA"))
        self.assertAlmostEqual(c["mean"], 4.0)
        table = au.best_table([c], au.aggregate(test, "accuracy"), "pretrained", "accuracy", "test")
        self.assertIn(r"\label{tab:augment-best-pretrained-test}", table)
        self.assertIn("ResNet-50 & 5 & 40.00 $\\pm$ 1.00 & TA", table)


if __name__ == "__main__":
    unittest.main()
