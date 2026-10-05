"""Tests for the Stable Diffusion pool generator, with a fake pipeline (no GPU, no model download)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.component.generate_sd import (
    TEMPLATES,
    class_phrase,
    downsample,
    generate_class,
    image_seed,
    merge_shards,
    prompt_for,
    save_shard,
    shard_is_complete,
)
from src.component.synthetic_pool import load_pool

SETTINGS = {"steps": 2, "guidance": 7.5, "size": 64}


class FakePipeline:
    """Returns one solid-colour image per prompt; the colour comes from the seed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        images = [Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(g % 256, 0, 0))
                  for g in kwargs["generator"]]
        return SimpleNamespace(images=images)


class PromptTest(unittest.TestCase):
    def test_ambiguous_names_are_rewritten(self) -> None:
        self.assertEqual(class_phrase("ray"), "stingray")
        self.assertEqual(class_phrase("maple_tree"), "maple tree")

    def test_article_follows_the_phrase(self) -> None:
        self.assertEqual(prompt_for("apple", 0), "a photo of an apple.")
        self.assertEqual(prompt_for("bear", 0), "a photo of a bear.")

    def test_identifier_names_the_adapted_class(self) -> None:
        self.assertEqual(prompt_for("apple", 0, "cfr"), "a photo of a cfr apple.")
        self.assertEqual(prompt_for("ray", 4, "cfr"), "a photo of a small cfr stingray.")

    def test_templates_repeat_in_order(self) -> None:
        self.assertEqual(prompt_for("bear", 1), prompt_for("bear", 1 + len(TEMPLATES)))
        self.assertNotEqual(prompt_for("bear", 0), prompt_for("bear", 1))


class ImageSeedTest(unittest.TestCase):
    def test_fixed_and_distinct(self) -> None:
        seeds = {image_seed(0, label, i) for label in range(20) for i in range(50)}
        self.assertEqual(len(seeds), 1000)
        self.assertEqual(image_seed(0, 3, 7), image_seed(0, 3, 7))
        self.assertNotEqual(image_seed(0, 3, 7), image_seed(1, 3, 7))


class DownsampleTest(unittest.TestCase):
    def test_returns_32x32_uint8_and_keeps_the_colour(self) -> None:
        small = downsample(Image.new("RGB", (512, 512), color=(10, 200, 30)))
        self.assertEqual((small.shape, small.dtype), ((32, 32, 3), np.uint8))
        self.assertEqual(small[5, 5].tolist(), [10, 200, 30])


class GenerateClassTest(unittest.TestCase):
    def run_class(self, pipe: FakePipeline, preview_dir: Path | None = None) -> dict:
        return generate_class(pipe, label=3, name="bear", per_class=5, generation_seed=0, batch_size=2,
                              settings=SETTINGS, make_generator=lambda s: s, preview_dir=preview_dir, preview=2)

    def test_batches_prompts_and_seeds(self) -> None:
        pipe = FakePipeline()
        shard = self.run_class(pipe)
        self.assertEqual([len(c["prompt"]) for c in pipe.calls], [2, 2, 1])
        self.assertEqual(pipe.calls[0]["prompt"], [prompt_for("bear", 0), prompt_for("bear", 1)])
        self.assertEqual(shard["seeds"].tolist(), [image_seed(0, 3, i) for i in range(5)])
        self.assertEqual(shard["images"].shape, (5, 32, 32, 3))
        self.assertEqual(shard["images"][4, 0, 0, 0], image_seed(0, 3, 4) % 256)

    def test_lora_prompts_and_negative_prompt(self) -> None:
        pipe = FakePipeline()
        generate_class(pipe, 3, "bear", 1, 0, 1, {**SETTINGS, "negative_prompt": "text"}, lambda s: s,
                       identifier="cfr")
        self.assertEqual(pipe.calls[0]["prompt"], ["a photo of a cfr bear."])
        self.assertEqual(pipe.calls[0]["negative_prompt"], ["text"])

    def test_saves_previews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.run_class(FakePipeline(), Path(tmp))
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["003_0.jpg", "003_1.jpg"])


class ShardsTest(unittest.TestCase):
    def test_merge_gives_a_sorted_pool_and_detects_missing_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for label in range(2):
                shard = generate_class(FakePipeline(), label, "bear", 3, 0, 2, SETTINGS, lambda s: s)
                save_shard(out, shard)
            self.assertTrue(shard_is_complete(out, 1, 3))
            self.assertFalse(shard_is_complete(out, 1, 4))
            with self.assertRaises(FileNotFoundError):
                merge_shards(out, num_classes=3, per_class=3)
            path, sha = merge_shards(out, num_classes=2, per_class=3)
            images, labels, index = load_pool(path)
        self.assertEqual(labels.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(index.tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(len(sha), 64)


if __name__ == "__main__":
    unittest.main()
