"""Tests for the Stage 3 results tables (rows of scores, no runs or torch)."""

import unittest

from src.component import stage3_results as s3


def rows(init: str, budget: str, scores: dict[str, list[float]], model: str = "resnet50",
         metric: str = "balanced_accuracy") -> list[dict]:
    return [{"init": init, "model": model, "budget": budget, "condition": c, "seed": s, metric: v,
             "accuracy": v + 10}
            for c, values in scores.items() for s, v in enumerate(values)]


class ChooseTest(unittest.TestCase):
    def test_best_conventional_by_validation_balanced_accuracy_with_three_seeds(self) -> None:
        data = rows("scratch", "5", {"randaugment": [10, 11, 12], "autoaugment": [12, 13, 14],
                                     "cutmix": [30, 31]})  # only 2 seeds
        self.assertEqual(s3.choose_conventional(data), {("scratch", "resnet50", "5"): "autoaugment"})

    def test_tie_keeps_the_earlier_condition(self) -> None:
        data = rows("scratch", "5", {"mixup": [10, 11, 12], "trivialaugment": [10, 11, 12]})
        self.assertEqual(s3.choose_conventional(data)[("scratch", "resnet50", "5")], "trivialaugment")


class TablesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = rows("pretrained", "2", {
            "real_only": [40, 41, 42], "randaugment": [44, 45, 46], "augmix": [45, 46, 47],
            "sd_prompt": [48, 49, 50], "sd_prompt_randaugment": [49, 50, 51], s3.OURS: [55, 56, 58]})
        self.data += rows("pretrained", "10", {"real_only": [60, 61, 62], "randaugment": [63, 64, 65],
                                               s3.OURS: [62, 64, 64]})
        self.best = s3.choose_conventional(self.data)

    def test_ours_is_compared_with_the_chosen_conventional(self) -> None:
        changes = s3.study_changes(self.data, "sd", "balanced_accuracy", self.best)
        vs_best = {c["budget"]: c for c in changes if c["control"] == s3.BEST}
        self.assertEqual(vs_best["2"]["best"], "augmix")
        self.assertAlmostEqual(vs_best["2"]["mean"], 10 + 1 / 3)
        self.assertEqual(vs_best["10"]["verdict"], "inconclusive")
        self.assertEqual([c["budget"] for c in changes if c["label"] == "SD$-$real"], ["2"])
        self.assertTrue(all("best" in c for c in changes))

    def test_tables(self) -> None:
        changes = s3.study_changes(self.data, "sd", "balanced_accuracy", self.best)
        table = s3.study_table(changes, "flowers102", "sd", "pretrained", "balanced_accuracy", "test")
        self.assertIn(r"\label{tab:stage3-flowers102-sd-pretrained-test}", table)
        self.assertIn("ResNet-50 & 2 &", table)
        self.assertLess(table.index("ResNet-50 & 2 &"), table.index("ResNet-50 & 10 &"))
        means = s3.aggregate(self.data, "balanced_accuracy")
        scores = s3.scores_table(means, changes, self.best, "flowers102", "pretrained", "balanced_accuracy",
                                 "validation")
        self.assertIn("ResNet-50 & 2 & 41.00 $\\pm$ 1.00 & AugMix & 46.00 $\\pm$ 1.00 & 49.00", scores)
        self.assertIn("Flowers-102", scores)

    def test_conventional_study(self) -> None:
        changes = s3.study_changes(self.data, "conventional", "accuracy", self.best)
        self.assertEqual({c["label"] for c in changes}, {"RA$-$real", "AugMix$-$real"})


if __name__ == "__main__":
    unittest.main()
