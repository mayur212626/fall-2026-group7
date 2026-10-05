"""Tests for the LoRA rank selection rule."""

import json
import tempfile
import unittest
from pathlib import Path

from src.component.choose_lora_rank import choose_rank, pilot_scores


class ChooseRankTest(unittest.TestCase):
    def test_default_rank_kept_within_margin(self) -> None:
        scores = {4: {"a": 70.0, "b": 80.0}, 8: {"a": 70.2, "b": 80.0}, 16: {"a": 70.5, "b": 80.3}}
        self.assertEqual(choose_rank(scores, margin=0.5)["chosen_rank"], 8)

    def test_clearly_better_rank_wins(self) -> None:
        scores = {4: {"a": 72.0, "b": 82.0}, 8: {"a": 70.0, "b": 80.0}, 16: {"a": 70.0, "b": 80.5}}
        self.assertEqual(choose_rank(scores, margin=0.5)["chosen_rank"], 4)

    def test_incomplete_ranks_are_ignored(self) -> None:
        scores = {4: {"a": 99.0}, 8: {"a": 70.0, "b": 80.0}}
        self.assertEqual(choose_rank(scores, margin=0.5)["chosen_rank"], 8)

    def test_requires_the_default_rank(self) -> None:
        with self.assertRaises(ValueError):
            choose_rank({4: {"a": 70.0, "b": 80.0}}, margin=0.5)

    def test_reads_rank_from_the_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "r16" / "pretrained_resnet50_sd_lora_b50_s100"
            run.mkdir(parents=True)
            (run / "config.json").write_text(json.dumps({"settings": {"model": "resnet50"}}))
            (run / "result.json").write_text(json.dumps({"best_validation": {"accuracy": 71.5}}))
            self.assertEqual(pilot_scores(Path(tmp)), {16: {"resnet50": 71.5}})


if __name__ == "__main__":
    unittest.main()
