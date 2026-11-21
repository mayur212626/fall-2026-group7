"""Tests for the Stage 3 datasets: preprocessing and split manifests (NumPy and Pillow only)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.component import stage3_data as sd


def fake_lesions(sizes: list[int], seed: int = 0) -> tuple[list[str], list[int]]:
    """Lesions of 1 to 3 images each, ``sizes[c]`` images in class ``c``."""
    rng = np.random.default_rng(seed)
    lesions, labels = [], []
    for label, size in enumerate(sizes):
        done = 0
        while done < size:
            count = min(size - done, int(rng.integers(1, 4)))
            lesions += [f"L{label}_{done}"] * count
            labels += [label] * count
            done += count
    return lesions, labels


class DatasetSpecTest(unittest.TestCase):
    def test_class_lists(self) -> None:
        self.assertEqual(len(set(sd.FLOWERS102_CLASSES)), 102)
        self.assertEqual(len(sd.HAM10000_CODES), len(sd.HAM10000_CLASSES), 7)
        self.assertEqual(sd.HAM10000_CODES, sorted(sd.HAM10000_CODES))

    def test_pools_cover_five_to_one_at_the_largest_budget(self) -> None:
        for spec in sd.DATASETS.values():
            self.assertGreaterEqual(spec["sd_per_class"], 5 * max(int(b) for b in spec["budgets"]))
            self.assertIn(spec["pilot_budget"], spec["budgets"])

    def test_templates_fill_in(self) -> None:
        for spec in sd.DATASETS.values():
            for template in spec["templates"]:
                text = template.format(article="a", name="rose")
                self.assertIn("rose", text)
                self.assertNotIn("{", text)


class PrepareImageTest(unittest.TestCase):
    def test_short_side_then_center_crop(self) -> None:
        image = Image.new("RGB", (600, 450), (0, 0, 255))
        image.paste((255, 0, 0), (0, 0, 60, 450))  # a red strip on the left, cropped away
        out = sd.prepare_image(image)
        self.assertEqual((out.shape, out.dtype), ((224, 224, 3), np.uint8))
        self.assertEqual(out[:, 0, 0].max(), 0)
        self.assertEqual(out[112, 112].tolist(), [0, 0, 255])

    def test_square_generated_image(self) -> None:
        self.assertEqual(sd.prepare_image(Image.new("RGB", (512, 512))).shape, (224, 224, 3))


class BudgetTest(unittest.TestCase):
    def test_budgets_are_nested(self) -> None:
        budgets = sd.nested_budgets({0: [5, 3, 9, 1], 1: [2, 8, 4, 7]}, ["1", "2", "4"])
        self.assertEqual(budgets, {"1": [2, 5], "2": [2, 3, 5, 8], "4": [1, 2, 3, 4, 5, 7, 8, 9]})

    def test_rejects_a_class_smaller_than_the_largest_budget(self) -> None:
        with self.assertRaises(ValueError):
            sd.nested_budgets({0: [1, 2], 1: [3]}, ["2"])


class FlowersManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = [c for c in range(3) for _ in range(30)]
        self.split_of = ["train"] * 10 + ["val"] * 10 + ["test"] * 10
        self.split_of = self.split_of * 3
        self.manifest = sd.flowers_manifest(self.split_of, self.labels, ["2", "5", "10"])

    def test_keeps_the_official_split(self) -> None:
        split = np.asarray(self.split_of)
        self.assertEqual(self.manifest["validation_indices"], np.flatnonzero(split == "val").tolist())
        self.assertEqual(self.manifest["test_indices"], np.flatnonzero(split == "test").tolist())
        self.assertEqual(self.manifest["train_indices"]["10"], np.flatnonzero(split == "train").tolist())

    def test_labels_must_start_at_zero(self) -> None:
        with self.assertRaises(ValueError):
            sd.check_manifest({"num_classes": 3, **self.manifest}, [label + 1 for label in self.labels])

    def test_budgets_are_balanced_nested_and_deterministic(self) -> None:
        manifest = {"num_classes": 3, **self.manifest}
        sd.check_manifest(manifest, self.labels)
        again = sd.flowers_manifest(self.split_of, self.labels, ["2", "5", "10"])
        self.assertEqual(again, self.manifest)


class LesionSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lesions, self.labels = fake_lesions([327, 514, 1099, 115, 1113, 1200, 142])
        self.manifest = sd.lesion_split(self.lesions, self.labels, 0.2, 0.1, ["5", "10", "20", "50"])

    def test_no_lesion_in_two_splits(self) -> None:
        sd.check_manifest({"num_classes": 7, "lesion_ids": self.lesions, **self.manifest}, self.labels)

    def test_fractions_per_class(self) -> None:
        labels = np.asarray(self.labels)
        for label in range(7):
            total = np.sum(labels == label)
            test = np.sum(labels[self.manifest["test_indices"]] == label)
            validation = np.sum(labels[self.manifest["validation_indices"]] == label)
            self.assertAlmostEqual(test / total, 0.2, delta=0.03)
            self.assertAlmostEqual(validation / total, 0.1, delta=0.03)

    def test_small_budgets_use_distinct_lesions(self) -> None:
        lesions = np.asarray(self.lesions)
        labels = np.asarray(self.labels)
        for label in range(7):
            chosen = [i for i in self.manifest["train_indices"]["20"] if labels[i] == label]
            self.assertEqual(len(set(lesions[chosen])), 20)

    def test_deterministic(self) -> None:
        self.assertEqual(sd.lesion_split(self.lesions, self.labels, 0.2, 0.1, ["5", "10", "20", "50"]), self.manifest)

    def test_a_lesion_with_two_classes_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            sd.lesion_split(["a", "a", "b"], [0, 1, 1], 0.2, 0.1, ["1"])

    def test_overlap_is_caught(self) -> None:
        broken = {"num_classes": 7, "lesion_ids": self.lesions, **self.manifest}
        broken["test_indices"] = broken["test_indices"] + broken["validation_indices"][:1]
        with self.assertRaises(ValueError):
            sd.check_manifest(broken, self.labels)


class HamFilesTest(unittest.TestCase):
    def test_reads_comma_and_tab_metadata_sorted_by_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name, sep in (("m.csv", ","), ("m.tab", "\t")):
                path = Path(tmp) / name
                path.write_text(sep.join(["lesion_id", "image_id", "dx"]) + "\n"
                                + sep.join(["HAM_2", "ISIC_0000002", "nv"]) + "\n"
                                + sep.join(['"HAM_1"', '"ISIC_0000001"', '"mel"']) + "\n")
                rows = sd.read_ham_metadata(path)
                self.assertEqual([r["image_id"] for r in rows], ["ISIC_0000001", "ISIC_0000002"])
                self.assertEqual(rows[0]["dx"], "mel")

    def test_prepare_reads_images_in_metadata_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            (raw / "part").mkdir()
            (raw / "HAM10000_metadata.csv").write_text(
                "lesion_id,image_id,dx\nHAM_1,ISIC_2,nv\nHAM_1,ISIC_1,nv\nHAM_2,ISIC_3,df\n")
            for i, color in ((1, 10), (2, 20), (3, 30)):
                Image.new("RGB", (600, 450), (color, 0, 0)).save(raw / "part" / f"ISIC_{i}.jpg", quality=100)
            images, labels, lesions, ids = sd.prepare_ham(raw)
        self.assertEqual(ids, ["ISIC_1", "ISIC_2", "ISIC_3"])
        self.assertEqual(labels, [sd.HAM10000_CODES.index("nv")] * 2 + [sd.HAM10000_CODES.index("df")])
        self.assertEqual(lesions, ["HAM_1", "HAM_1", "HAM_2"])
        self.assertEqual(images.shape, (3, 224, 224, 3))
        self.assertAlmostEqual(float(images[2, :, :, 0].mean()), 30, delta=2)


class ManifestJsonTest(unittest.TestCase):
    def test_one_line_per_key_and_valid_json(self) -> None:
        manifest = {"dataset": "HAM10000", "train_indices": {"5": [1, 2]}, "test_indices": [3, 4]}
        text = sd.manifest_json(manifest)
        self.assertEqual(json.loads(text), manifest)
        self.assertEqual(len(text.splitlines()), 5)


class ArraysTest(unittest.TestCase):
    def test_round_trip_memory_mapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = np.random.default_rng(0).integers(0, 256, (4, 8, 8, 3), dtype=np.uint8)
            sd.write_arrays(Path(tmp), images, [0, 1, 0, 1])
            loaded, labels = sd.load_images(Path(tmp))
            self.assertIsInstance(loaded, np.memmap)
            np.testing.assert_array_equal(loaded, images)
            self.assertEqual(labels, [0, 1, 0, 1])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["images.npy", "labels.npy"])
            del loaded


class ConfigTest(unittest.TestCase):
    def test_stage3_configs_have_every_budget_and_both_models(self) -> None:
        for dataset, spec in sd.DATASETS.items():
            for init in ("pretrained", "scratch"):
                path = Path("src/component/configs") / f"train_{init}_{dataset}.json"
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["dataset"], dataset)
                self.assertEqual(sorted(config["steps"], key=int), spec["budgets"])
                self.assertEqual(sorted(config["models"]), ["resnet50", "vit_b_16"])


if __name__ == "__main__":
    unittest.main()
