"""Tests for the SD img2img pool generator, with a fake pipeline (no GPU, no model download)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.component.generate_img2img import generate_class, registry_fields, source_order, upsample
from src.component.generate_sd import merge_shards, prompt_for, save_shard
from src.component.generator_registry import FIELDS
from src.component.synthetic_pool import load_pool, select_per_class

SETTINGS = {"steps": 4, "guidance": 7.5, "strength": 0.5, "size": 64}


class FakeImg2Img:
    """Returns each input image with its red channel set from the seed, so edits stay traceable."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        out = []
        for image, seed in zip(kwargs["image"], kwargs["generator"], strict=True):
            array = np.asarray(image).copy()
            array[..., 0] = seed % 256
            out.append(Image.fromarray(array))
        return SimpleNamespace(images=out)


def manifest(labels: np.ndarray, per_class: dict[str, int]) -> dict[str, list[int]]:
    """Nested budgets: budget k takes the first k images of each class (by index)."""
    out = {}
    for budget, k in per_class.items():
        out[budget] = sorted(i for c in np.unique(labels) for i in np.flatnonzero(labels == c)[:k])
    return out


class SourceOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.labels = rng.permutation(np.repeat(np.arange(3), 8))
        self.budgets = ("2", "4", "8")
        self.indices = manifest(self.labels, {"2": 2, "4": 4, "8": 8})

    def test_first_k_sources_are_exactly_budget_k(self) -> None:
        order = source_order(self.indices, self.labels, 3, self.budgets)
        for budget in self.budgets:
            k = int(budget)
            chosen = sorted(i for c in range(3) for i in order[c][:k])
            self.assertEqual(chosen, self.indices[budget])
        for c in range(3):
            self.assertTrue(all(self.labels[i] == c for i in order[c]))

    def test_rejects_budgets_that_are_not_nested_or_short(self) -> None:
        broken = dict(self.indices)
        broken["4"] = broken["4"][1:] + [broken["8"][-1]]
        with self.assertRaises(ValueError):
            source_order(broken, self.labels, 3, self.budgets)
        mislabelled = {**self.indices, "9": self.indices["8"]}  # named 9 per class but has 8
        with self.assertRaises(ValueError):
            source_order(mislabelled, self.labels, 3, ("2", "4", "9"))


class GenerateClassTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.sources = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
        self.pipe = FakeImg2Img()

    def test_edits_every_source_in_batches_with_the_class_prompts(self) -> None:
        shard = generate_class(self.pipe, 7, "aquarium_fish", self.sources, 0, 2, SETTINGS, lambda s: s)
        self.assertEqual(shard["images"].shape, (5, 32, 32, 3))
        self.assertEqual(shard["count"], 5)
        self.assertEqual([len(c["prompt"]) for c in self.pipe.calls], [2, 2, 1])
        self.assertEqual(self.pipe.calls[0]["prompt"][1], prompt_for("aquarium_fish", 1))
        self.assertTrue(all(c["strength"] == 0.5 for c in self.pipe.calls))
        self.assertEqual(self.pipe.calls[0]["image"][0].size, (64, 64))
        self.assertEqual(len(set(shard["seeds"].tolist())), 5)

    def test_each_edit_comes_from_its_own_source(self) -> None:
        # solid colours: the fake keeps green and blue, which survive the up- and downsampling exactly
        sources = np.zeros((5, 32, 32, 3), dtype=np.uint8)
        sources[..., 1] = np.arange(5)[:, None, None] * 40
        sources[..., 2] = 200 - np.arange(5)[:, None, None] * 30
        shard = generate_class(self.pipe, 3, "bear", sources, 0, 4, SETTINGS, lambda s: s)
        np.testing.assert_array_equal(shard["images"][..., 1:], sources[..., 1:])

    def test_previews_show_source_and_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generate_class(self.pipe, 3, "bear", self.sources, 0, 4, SETTINGS, lambda s: s, Path(tmp), preview=2)
            files = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(files, ["003_0.jpg", "003_1.jpg"])
            self.assertEqual(Image.open(Path(tmp) / files[0]).size, (128, 64))


class PoolTest(unittest.TestCase):
    def test_pool_selection_matches_the_budget(self) -> None:
        pipe, rng = FakeImg2Img(), np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as tmp:
            for label in range(2):
                sources = rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
                save_shard(Path(tmp), generate_class(pipe, label, "bear", sources, 0, 4, SETTINGS, lambda s: s))
            path, _ = merge_shards(Path(tmp), 2, 4)
            images, labels, index = load_pool(path)
        rows = select_per_class(labels, index, 2, 2)
        self.assertEqual(index[rows].tolist(), [0, 1, 0, 1])  # budget 2 uses edits 0 and 1 of each class


class RegistryTest(unittest.TestCase):
    def test_fields_are_registry_columns(self) -> None:
        meta = {"model": "m", "revision": "r", "strength": 0.5, "per_class": 50, "source_budgets": ["5", "50"],
                "scheduler": "DPM", "size": 512, "output_size": 32, "resize": "bicubic", "steps": 25,
                "guidance": 7.5, "templates": ["a photo of {article} {name}."], "negative_prompt": "text",
                "generation_seed": 0, "seed_rule": "rule", "generation_seconds": 120.0,
                "finished": "2026-10-28T10:00:00+00:00", "pool_sha256": "abc123def4567890"}
        fields = registry_fields(meta)
        self.assertLessEqual(set(fields), set(FIELDS))
        self.assertEqual(fields["condition"], "sd_img2img")
        self.assertEqual(fields["generation_minutes"], 2.0)


class UpsampleTest(unittest.TestCase):
    def test_returns_rgb_of_the_pipeline_size(self) -> None:
        image = upsample(np.zeros((32, 32, 3), dtype=np.uint8), 64)
        self.assertEqual((image.size, image.mode), ((64, 64), "RGB"))


if __name__ == "__main__":
    unittest.main()
