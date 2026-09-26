"""Tests for classifier construction. No pretrained weights are downloaded."""

import unittest

import torch

from src.component.models import build_model, weights_name


class BuildModelTest(unittest.TestCase):
    def test_resnet50_has_requested_output_size(self) -> None:
        model = build_model("resnet50", pretrained=False, num_classes=100).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 3, 32, 32)).shape), (2, 100))

    def test_vit_b_16_has_requested_output_size(self) -> None:
        model = build_model("vit_b_16", pretrained=False, num_classes=100).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(1, 3, 224, 224)).shape), (1, 100))

    def test_same_seed_gives_same_initial_weights(self) -> None:
        torch.manual_seed(0)
        first = build_model("resnet50", pretrained=False, num_classes=100)
        torch.manual_seed(0)
        second = build_model("resnet50", pretrained=False, num_classes=100)
        self.assertTrue(torch.equal(first.fc.weight, second.fc.weight))
        self.assertTrue(torch.equal(first.conv1.weight, second.conv1.weight))

    def test_rejects_unknown_model(self) -> None:
        with self.assertRaises(ValueError):
            build_model("resnet18", pretrained=False, num_classes=100)


class WeightsNameTest(unittest.TestCase):
    def test_names_pretrained_weights(self) -> None:
        self.assertEqual(weights_name("resnet50", pretrained=True), "IMAGENET1K_V2")
        self.assertEqual(weights_name("vit_b_16", pretrained=True), "IMAGENET1K_V1")

    def test_scratch_has_no_weights(self) -> None:
        self.assertIsNone(weights_name("resnet50", pretrained=False))


if __name__ == "__main__":
    unittest.main()
