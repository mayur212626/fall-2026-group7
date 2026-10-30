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


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quality_file(folder: Path) -> Path:
    # fid, floor, precision, recall: class prompts more precise, LoRA lower FID (except b5) and higher recall
    values = {"5": ((103.0, 104.8), 88.9, (0.76, 0.73), (0.43, 0.50)),
              "10": ((71.9, 71.2), 54.3, (0.75, 0.74), (0.36, 0.42)),
              "full": ((24.7, 21.2), "", (0.66, 0.64), (0.23, 0.29))}
    rows = []
    for budget, (fids, floor, precs, recs) in values.items():
        for prefix, fid, prec, rec in zip(("sd15_prompt", "sd15_lora"), fids, precs, recs, strict=True):
            rows.append({"name": f"{prefix}_b{budget}", "reference": "pool, class-balanced, seed 0", "fid": fid,
                         "fid_real_floor": floor, "precision": prec, "recall": rec})
    rows.append({"name": "something_else_b5", "fid": 1, "precision": 1, "recall": 1})
    path = folder / "generator_quality.csv"
    write_csv(path, QUALITY_FIELDS, rows)
    return path


def gain_files(folder: Path, init: str, prompt: dict, lora: dict) -> None:
    """prompt / lora: {(model, budget): {seed: change}}."""
    fields = ["treatment", "control", "init", "model", "budget", "seeds", "changes", "mean", "sd", "verdict"]
    for generator, values in (("sd_prompt", prompt), ("sd_lora", lora)):
        rows = [{"treatment": generator, "control": "real_only", "init": init, "model": m, "budget": b,
                 "seeds": " ".join(map(str, v)), "changes": " ".join(map(str, v.values()))}
                for (m, b), v in values.items()]
        write_csv(folder / pu.GAIN_FILES[generator].format(init=init), fields, rows)


class ReadTest(unittest.TestCase):
    def test_quality_maps_names_to_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = pu.read_quality(quality_file(Path(tmp)))
        self.assertEqual(set(q), {(g, b) for g in ("sd_prompt", "sd_lora") for b in ("5", "10", "full")})
        self.assertEqual(q[("sd_lora", "5")], {"fid": 104.8, "precision": 0.73, "recall": 0.5})

    def test_gains_are_per_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gain_files(Path(tmp), "pretrained", {("resnet50", "5"): {0: 1.0, 1: 2.0, 3: 4.0}}, {})
            g = pu.read_gains(Path(tmp), "pretrained")
        self.assertEqual(g[("resnet50", "sd_prompt", "5")], {0: 1.0, 1: 2.0, 3: 4.0})


class StatsTest(unittest.TestCase):
    def test_binomial(self) -> None:
        self.assertAlmostEqual(pu.binomial_two_sided(10, 10), 2 / 1024)
        self.assertAlmostEqual(pu.binomial_two_sided(5, 10), 1.0)
        self.assertTrue(np.isnan(pu.binomial_two_sided(0, 0)))

    def test_holm(self) -> None:
        np.testing.assert_allclose(pu.holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])


class ComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.quality = {("sd_prompt", "5"): {"fid": 100.0, "precision": 0.8, "recall": 0.4},
                        ("sd_lora", "5"): {"fid": 90.0, "precision": 0.7, "recall": 0.5},
                        ("sd_prompt", "10"): {"fid": 70.0, "precision": 0.8, "recall": 0.4},
                        ("sd_lora", "10"): {"fid": 70.0, "precision": 0.7, "recall": 0.5}}
        self.measures = {"fid": False, "precision": True, "recall": True}

    def test_agreement_uses_the_direction_and_shared_seeds(self) -> None:
        gains = {"pretrained": {("resnet50", "sd_prompt", "5"): {0: 10.0, 1: 12.0, 3: 50.0},
                                ("resnet50", "sd_lora", "5"): {0: 8.0, 1: 13.0}}}  # seed 3 only for prompts
        [row] = pu.comparisons(self.quality, gains, self.measures)
        self.assertEqual(row["seeds"], "0 1")
        self.assertAlmostEqual(row["gain_difference"], 0.5)  # (2 - 1) / 2, seed 3 left out
        self.assertEqual(row["seeds_prompt_ahead"], 1)
        self.assertEqual(row["agree_fid"], 0)        # LoRA has lower FID but prompts gain more
        self.assertEqual(row["agree_precision"], 1)  # prompts more precise and gain more
        self.assertEqual(row["agree_recall"], 0)

    def test_ties_and_missing_sets_are_left_out(self) -> None:
        gains = {"scratch": {("vit_b_16", "sd_prompt", "10"): {0: 2.0}, ("vit_b_16", "sd_lora", "10"): {0: 1.0},
                             ("vit_b_16", "sd_prompt", "20"): {0: 2.0}}}
        [row] = pu.comparisons(self.quality, gains, self.measures)
        self.assertEqual(row["agree_fid"], "")  # equal FID

    def test_summary_counts_per_init_and_overall(self) -> None:
        rows = [{"init": "pretrained", "agree_m": 1}, {"init": "pretrained", "agree_m": 1},
                {"init": "scratch", "agree_m": 0}, {"init": "scratch", "agree_m": ""}]
        stats = {(s["measure"], s["init"]): s for s in pu.summary(rows, {"m": True})}
        self.assertEqual((stats[("m", "all")]["agree"], stats[("m", "all")]["comparisons"]), (2, 3))
        self.assertEqual((stats[("m", "scratch")]["agree"], stats[("m", "scratch")]["comparisons"]), (0, 1))
        self.assertEqual(stats[("m", "scratch")]["p_holm"], "")
        self.assertIn(r"\label{tab:pool-utility}", pu.summary_table(list(stats.values())))

    @staticmethod
    def rows_for(verdicts: dict[str, tuple[float, float]], agree: list[int]) -> list[dict]:
        """Two cells (pretrained, scratch) per budget sharing the budget's measure values."""
        rows, votes = [], iter(agree)
        for budget, (prompt, lora) in verdicts.items():
            for init in ("pretrained", "scratch"):
                rows.append({"init": init, "budget": budget, "m_sd_prompt": prompt, "m_sd_lora": lora,
                             "agree_m": next(votes)})
        return rows

    def test_constant_verdict_gets_no_p_value(self) -> None:
        # the measure prefers prompts at every budget: its count only says how often prompts win
        rows = self.rows_for({"5": (0.8, 0.7), "10": (0.8, 0.7), "20": (0.8, 0.7)}, [1, 1, 1, 1, 1, 1])
        stats = {(s["measure"], s["init"]): s for s in pu.summary(rows, {"m": True})}
        overall = stats[("m", "all")]
        self.assertEqual((overall["budgets_prompt_better"], overall["budgets"], overall["verdict_changes"]), (3, 3, 0))
        self.assertEqual((overall["agree"], overall["comparisons"]), (6, 6))
        self.assertTrue(np.isnan(overall["p_value"]))
        self.assertTrue(np.isnan(overall["p_holm"]))
        table = pu.summary_table(list(stats.values()), "0--2")
        self.assertIn("m & 3/3 & 3/3 & 3/3 & 6/6 & -- & --", table)
        self.assertIn("seeds 0--2", table)

    def test_changing_verdict_is_tested(self) -> None:
        # prompts better at 5 only (lower is better), so the verdict changes and a p-value is given
        rows = self.rows_for({"5": (90.0, 95.0), "10": (80.0, 70.0), "20": (60.0, 50.0)}, [1, 0, 0, 0, 0, 1])
        overall = {(s["measure"], s["init"]): s for s in pu.summary(rows, {"m": False})}[("m", "all")]
        self.assertEqual((overall["budgets_prompt_better"], overall["budgets"], overall["verdict_changes"]), (1, 3, 1))
        self.assertAlmostEqual(overall["p_value"], pu.binomial_two_sided(2, 6))
        self.assertAlmostEqual(overall["p_holm"], overall["p_value"])  # the only measure tested

    def test_seed_range(self) -> None:
        self.assertEqual(pu.seed_range(["0", "1", "2", "1"]), "0--2")
        self.assertEqual(pu.seed_range(["0", "2"]), "0, 2")
        self.assertEqual(pu.seed_range([]), "")


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
    def test_main_writes_tables_and_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tables, figures = Path(tmp) / "tables", Path(tmp) / "fig"
            tables.mkdir()
            quality_file(tables)
            seeds = {0: 0.0, 1: 0.0, 2: 0.0}
            prompt = {(m, b): {s: g for s in seeds} for m in pu.MODEL_NAMES for b, g in (("5", 12.0), ("10", 8.0),
                                                                                       ("full", 0.5))}
            lora = {(m, b): {s: g for s in seeds} for m in pu.MODEL_NAMES for b, g in (("5", 10.0), ("10", 6.0),
                                                                                     ("full", 0.0))}
            gain_files(tables, "pretrained", prompt, lora)
            with mock.patch.object(sys, "argv", ["pool_utility", "--tables", str(tables), "--figures", str(figures)]):
                pu.main()
            stats = {(r["measure"], r["init"]): r for r in pu.read_rows(tables / "pool_utility_summary.csv")}
            # prompts always gain more and are always more precise; LoRA has lower FID at 10 and full only
            self.assertEqual((stats[("precision", "all")]["agree"], stats[("precision", "all")]["comparisons"]),
                             ("6", "6"))
            self.assertEqual(stats[("fid", "all")]["agree"], "2")
            self.assertEqual(stats[("recall", "all")]["agree"], "0")
            # FID changes its verdict (prompts better at 5 only), precision and recall never do
            self.assertEqual(stats[("fid", "all")]["verdict_changes"], "1")
            self.assertEqual(stats[("precision", "all")]["p_value"], "nan")
            self.assertNotIn("fid_gap", {r["measure"] for r in stats.values()})
            self.assertTrue((figures / "pool_utility_measure_vs_gain.pdf").exists())
            self.assertTrue((tables / "pool_utility_summary.tex").exists())


if __name__ == "__main__":
    unittest.main()
