"""Tests for the generator registry: updating rows and the values recorded from pool.json."""

import tempfile
import unittest
from pathlib import Path

from src.component.generator_registry import FIELDS, generation_fields, read_registry, set_row

POOL = {
    "model": "stable-diffusion-v1-5/stable-diffusion-v1-5", "revision": "abc123", "scheduler": "DPMSolverMultistepScheduler",
    "steps": 25, "guidance": 7.5, "size": 512, "output_size": 32, "resize": "PIL bicubic (antialiased)",
    "templates": ["a photo of {article} {name}."], "identifier": None, "lora": None, "negative_prompt": "text",
    "generation_seed": 0, "seed_rule": "SeedSequence", "per_class": 450, "generation_seconds": 6000.0,
    "finished": "2026-09-30T03:00:00+00:00", "pool_sha256": "3e9d6b112c27de63",
}
LORA = {"rank": 8, "steps": 1000, "batch_size": 8, "lr": 1e-4, "identifier": "cfr", "budget": "5", "train_images": 500}


class SetRowTest(unittest.TestCase):
    def test_updates_one_row_and_keeps_the_others(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.csv"
            set_row(path, "sd_prompt", {"status": "planned", "ticket": "#9"})
            set_row(path, "lora_sd_b5", {"status": "planned", "ticket": "#15"})
            set_row(path, "sd_prompt", {"status": "generated", "fid": 24.66})
            rows = read_registry(path)
        self.assertEqual([r["generator"] for r in rows], ["sd_prompt", "lora_sd_b5"])
        self.assertEqual((rows[0]["status"], rows[0]["ticket"], rows[0]["fid"]), ("generated", "#9", "24.66"))
        self.assertEqual(rows[1]["status"], "planned")
        self.assertEqual(list(rows[0]), FIELDS)

    def test_rejects_unknown_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "unknown"):
            set_row(Path(tmp) / "registry.csv", "sd_prompt", {"gpu": "A10G"})


class GenerationFieldsTest(unittest.TestCase):
    def test_class_prompt_pool(self) -> None:
        fields = generation_fields(POOL)
        self.assertEqual((fields["condition"], fields["adaptation"], fields["adaptation_data"]), ("sd_prompt", "none", "none"))
        self.assertEqual((fields["revision"], fields["steps"], fields["guidance"]), ("abc123", 25, 7.5))
        self.assertEqual((fields["images_per_class"], fields["generation_minutes"]), (450, 100.0))
        self.assertIn("negative: text", fields["prompt_template"])
        self.assertTrue(fields["status"].startswith("generated 2026-09-30"))
        self.assertEqual(set(fields) - set(FIELDS), set())

    def test_lora_pool_records_the_adapter_and_its_budget(self) -> None:
        fields = generation_fields({**POOL, "lora": LORA, "identifier": "cfr", "per_class": 5})
        self.assertEqual(fields["condition"], "sd_lora")
        self.assertIn("LoRA rank 8", fields["adaptation"])
        self.assertEqual(fields["adaptation_data"], "training images of budget 5 (500 images)")
        self.assertIn("'cfr <class>'", fields["prompt_template"])


if __name__ == "__main__":
    unittest.main()
