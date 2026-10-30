"""Tests for the pool-level RQ3 analysis (CSV files only)."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.component import pool_utility as pu

QUALITY_FIELDS = ["name", "pool_sha256", "images", "per_class", "reference", "reference_images", "fid", "fid_sd",
                  "fid_real_floor", "precision", "recall", "k", "features", "measured"]
PREFIXES = {"sd_prompt": "sd15_prompt", "sd_lora": "sd15_lora", "sd_img2img": "sd15_img2img"}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quality_file(path: Path, values: dict[tuple[str, str], tuple[float, float, float]]) -> Path:
    """values: {(set, budget): (fid, precision, recall)}, plus a row of an unknown generator."""
    rows = [{"name": f"{PREFIXES[s]}_b{b}", "fid": fid, "precision": prec, "recall": rec}
            for (s, b), (fid, prec, rec) in values.items()]
    rows.append({"name": "something_else_b5", "fid": 1, "precision": 1, "recall": 1})
    write_csv(path, QUALITY_FIELDS, rows)
    return path


def gain_files(folder: Path, init: str, gains: dict[str, dict]) -> None:
    """gains: {set: {(model, budget): {seed: change}}}."""
    fields = ["treatment", "control", "init", "model", "budget", "seeds", "changes", "mean", "sd", "verdict"]
    for name, values in gains.items():
        rows = [{"treatment": name, "control": "real_only", "init": init, "model": m, "budget": b,
                 "seeds": " ".join(map(str, v)), "changes": " ".join(map(str, v.values()))}
                for (m, b), v in values.items()]
        write_csv(folder / pu.GAIN_FILES[name].format(init=init), fields, rows)


class ReadTest(unittest.TestCase):
    def test_later_quality_tables_override_earlier_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = quality_file(Path(tmp) / "a.csv", {("sd_lora", "5"): (104.8, 0.73, 0.5),
                                                       ("sd_lora", "full"): (21.2, 0.64, 0.29)})
            second = quality_file(Path(tmp) / "b.csv", {("sd_lora", "5"): (105.4, 0.72, 0.5),
                                                        ("sd_img2img", "5"): (95.5, 0.73, 0.64)})
            q = pu.read_quality([first, second])
        self.assertEqual(set(q), {("sd_lora", "5"), ("sd_lora", "full"), ("sd_img2img", "5")})
        self.assertEqual(q[("sd_lora", "5")]["fid"], 105.4)   # from the second table
        self.assertEqual(q[("sd_lora", "full")]["fid"], 21.2)  # only in the first

    def test_gains_are_per_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gain_files(Path(tmp), "pretrained", {"sd_img2img": {("resnet50", "5"): {0: 1.0, 1: 2.0, 3: 4.0}}})
            g = pu.read_gains(Path(tmp), "pretrained")
        self.assertEqual(g[("resnet50", "sd_img2img", "5")], {0: 1.0, 1: 2.0, 3: 4.0})


class StatsTest(unittest.TestCase):
    def test_binomial(self) -> None:
        self.assertAlmostEqual(pu.binomial_two_sided(10, 10), 2 / 1024)
        self.assertAlmostEqual(pu.binomial_two_sided(5, 10), 1.0)
        self.assertTrue(np.isnan(pu.binomial_two_sided(0, 0)))

    def test_holm(self) -> None:
        np.testing.assert_allclose(pu.holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])

    def test_seed_range(self) -> None:
        self.assertEqual(pu.seed_range(["0", "1", "2", "1"]), "0--2")
        self.assertEqual(pu.seed_range(["0", "2"]), "0, 2")
        self.assertEqual(pu.seed_range([]), "")


class ComparisonTest(unittest.TestCase):
    measures = {"fid": False, "precision": True, "recall": True}

    def test_every_pair_with_direction_and_shared_seeds(self) -> None:
        quality = {("sd_prompt", "5"): {"fid": 100.0, "precision": 0.8, "recall": 0.4},
                   ("sd_lora", "5"): {"fid": 90.0, "precision": 0.7, "recall": 0.5},
                   ("sd_img2img", "5"): {"fid": 80.0, "precision": 0.75, "recall": 0.6}}
        gains = {"pretrained": {("resnet50", "sd_prompt", "5"): {0: 10.0, 1: 12.0, 3: 50.0},
                                ("resnet50", "sd_lora", "5"): {0: 8.0, 1: 13.0},
                                ("resnet50", "sd_img2img", "5"): {0: 5.0, 1: 6.0}}}
        rows = {r["pair"]: r for r in pu.comparisons(quality, gains, self.measures)}
        self.assertEqual(set(rows), {"sd_prompt-sd_lora", "sd_prompt-sd_img2img", "sd_lora-sd_img2img"})
        pl = rows["sd_prompt-sd_lora"]
        self.assertEqual(pl["seeds"], "0 1")                # seed 3 only for prompts
        self.assertAlmostEqual(pl["gain_difference"], 0.5)  # (2 - 1) / 2
        self.assertEqual(pl["seeds_a_ahead"], 1)
        self.assertEqual((pl["agree_fid"], pl["agree_precision"], pl["agree_recall"]), (0, 1, 0))
        li = rows["sd_lora-sd_img2img"]                     # LoRA gains more, img2img has lower FID, more recall
        self.assertAlmostEqual(li["gain_difference"], 5.0)
        self.assertEqual((li["agree_fid"], li["agree_precision"], li["agree_recall"]), (0, 0, 0))

    def test_ties_and_missing_sets_are_left_out(self) -> None:
        quality = {("sd_prompt", "10"): {"fid": 70.0, "precision": 0.8, "recall": 0.4},
                   ("sd_lora", "10"): {"fid": 70.0, "precision": 0.7, "recall": 0.5}}
        gains = {"scratch": {("vit_b_16", "sd_prompt", "10"): {0: 2.0}, ("vit_b_16", "sd_lora", "10"): {0: 1.0},
                             ("vit_b_16", "sd_img2img", "10"): {0: 1.0}, ("vit_b_16", "sd_prompt", "20"): {0: 2.0}}}
        [row] = pu.comparisons(quality, gains, self.measures)  # img2img has no quality, budget 20 has one set
        self.assertEqual(row["pair"], "sd_prompt-sd_lora")
        self.assertEqual(row["agree_fid"], "")  # equal FID

    @staticmethod
    def rows_for(values: dict[tuple[str, str], tuple[float, float]], agree: list[int]) -> list[dict]:
        """Two cells (pretrained, scratch) per (pair, budget) sharing that budget's measure values."""
        rows, votes = [], iter(agree)
        for (pair, budget), (a, b) in values.items():
            for init in ("pretrained", "scratch"):
                rows.append({"init": init, "budget": budget, "pair": pair, "m_a": a, "m_b": b, "agree_m": next(votes)})
        return rows

    def test_only_pairs_whose_verdict_changes_are_tested(self) -> None:
        # prompt-lora: the measure prefers prompts at every budget (not informative);
        # lora-img2img: LoRA better at 5, img2img at 10 and 20 (informative)
        values = {("sd_prompt-sd_lora", "5"): (0.8, 0.7), ("sd_prompt-sd_lora", "10"): (0.8, 0.7),
                  ("sd_prompt-sd_lora", "20"): (0.8, 0.7), ("sd_lora-sd_img2img", "5"): (0.7, 0.6),
                  ("sd_lora-sd_img2img", "10"): (0.6, 0.7), ("sd_lora-sd_img2img", "20"): (0.6, 0.7)}
        rows = self.rows_for(values, [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])
        stats = {(s["measure"], s["init"]): s for s in pu.summary(rows, {"m": True})}
        overall = stats[("m", "all")]
        self.assertEqual((overall["agree"], overall["comparisons"]), (9, 12))
        self.assertEqual((overall["informative_agree"], overall["informative"]), (3, 6))
        self.assertEqual(overall["varying_pairs"], "sd_lora-sd_img2img")
        self.assertEqual(overall["sd_prompt-sd_lora_first_better"], "3/3")
        self.assertEqual(overall["sd_lora-sd_img2img_first_better"], "1/3")
        self.assertEqual(overall["sd_prompt-sd_img2img_first_better"], "")
        self.assertAlmostEqual(overall["p_value"], pu.binomial_two_sided(3, 6))
        self.assertAlmostEqual(overall["p_holm"], overall["p_value"])  # the only measure tested
        self.assertEqual(stats[("m", "scratch")]["p_holm"], "")
        table = pu.summary_table(list(stats.values()), "0--2")
        self.assertIn(r"\label{tab:pool-utility}", table)
        self.assertIn("m & 3/3 & -- & 1/3 & 5/6 & 4/6 & 9/12 & 3/6 & 1.000 & 1.000", table)
        self.assertIn("seeds 0--2", table)

    def test_constant_verdicts_get_no_p_value(self) -> None:
        values = {("sd_prompt-sd_lora", "5"): (0.8, 0.7), ("sd_prompt-sd_lora", "10"): (0.8, 0.7)}
        overall = {(s["measure"], s["init"]): s for s in pu.summary(self.rows_for(values, [1, 1, 0, 1]),
                                                                  {"m": True})}[("m", "all")]
        self.assertEqual(overall["informative"], 0)
        self.assertTrue(np.isnan(overall["p_value"]))
        self.assertTrue(np.isnan(overall["p_holm"]))


class ExtraTest(unittest.TestCase):
    def test_extra_measures_need_a_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extra.csv"
            write_csv(path, ["generator", "budget", "kid"], [{"generator": "sd_lora", "budget": "5", "kid": 0.02}])
            q = {}
            self.assertEqual(pu.add_extra(q, path, {"kid": False}), ["kid"])
            self.assertAlmostEqual(q[("sd_lora", "5")]["kid"], 0.02)
            with self.assertRaises(ValueError):
                pu.add_extra({}, path, {})


class EndToEndTest(unittest.TestCase):
    def test_main_reads_both_quality_tables_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tables, figures = Path(tmp) / "tables", Path(tmp) / "fig"
            tables.mkdir()
            # whole-pool reference: full only matters; the budget-free table overrides 5 and 10
            quality_file(tables / "generator_quality.csv", {
                ("sd_prompt", "5"): (1.0, 0.0, 0.0), ("sd_lora", "5"): (1.0, 0.0, 0.0),
                ("sd_prompt", "full"): (24.7, 0.66, 0.23), ("sd_lora", "full"): (21.2, 0.64, 0.29)})
            quality_file(tables / "generator_quality_without_budget.csv", {
                ("sd_prompt", "5"): (106.6, 0.77, 0.41), ("sd_lora", "5"): (105.4, 0.72, 0.50),
                ("sd_img2img", "5"): (95.5, 0.73, 0.64), ("sd_prompt", "10"): (72.5, 0.77, 0.37),
                ("sd_lora", "10"): (70.7, 0.75, 0.40), ("sd_img2img", "10"): (61.4, 0.74, 0.61)})
            seeds = (0, 1, 2)
            gains = {"sd_prompt": {"5": 12.0, "10": 8.0, "full": 0.5}, "sd_lora": {"5": 10.0, "10": 6.0, "full": 0.0},
                     "sd_img2img": {"5": 6.0, "10": 5.0}}
            gain_files(tables, "pretrained", {name: {(m, b): {s: g for s in seeds} for m in pu.MODEL_NAMES
                                                     for b, g in values.items()} for name, values in gains.items()})
            with mock.patch.object(sys, "argv", ["pool_utility", "--tables", str(tables), "--figures", str(figures)]):
                pu.main()
            stats = {(r["measure"], r["init"]): r for r in pu.read_rows(tables / "pool_utility_summary.csv")}
            rows = pu.read_rows(tables / "pool_utility_comparisons.csv")
            self.assertEqual(len(rows), 2 * (3 + 3 + 1))  # two models; three pairs at 5 and 10, one at full
            # usefulness: prompts > LoRA > img2img everywhere; FID and recall rank them the other way round
            self.assertEqual((stats[("fid", "all")]["agree"], stats[("fid", "all")]["comparisons"]), ("0", "14"))
            self.assertEqual(stats[("recall", "all")]["agree"], "0")
            self.assertEqual(stats[("fid", "all")]["p_value"], "nan")
            # precision: LoRA vs img2img flips (img2img at 5, LoRA at 10), so those 4 comparisons are tested
            self.assertEqual((stats[("precision", "all")]["informative_agree"],
                              stats[("precision", "all")]["informative"]), ("2", "4"))
            self.assertTrue((figures / "pool_utility_measure_vs_gain.pdf").exists())
            self.assertTrue((tables / "pool_utility_summary.tex").exists())


if __name__ == "__main__":
    unittest.main()
