import json
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2

from scripts.train_classifier import evaluate, image_transform


class TrainingTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "configs/cifar100_training.json"
        self.config = json.loads(path.read_text())

    def test_clean_preprocessing(self):
        image = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
        transform = image_transform(self.config)
        first, second = transform(image), transform(image)
        self.assertEqual(first.shape, (3, 224, 224))
        self.assertEqual(first.dtype, torch.float32)
        torch.testing.assert_close(first, torch.ones_like(first))
        torch.testing.assert_close(first, second)

    def test_randaugment_output(self):
        image = Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8))
        torch.manual_seed(0)
        output = image_transform(self.config, randaugment=True)(image)
        self.assertEqual(output.shape, (3, 224, 224))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(output.min().item(), -1)
        self.assertLessEqual(output.max().item(), 1)

    def test_mixed_labels_and_cross_entropy_gradient(self):
        images = torch.arange(4).float().reshape(4, 1, 1, 1).expand(4, 3, 32, 32)
        labels = torch.arange(4)
        for policy in (v2.MixUp(alpha=self.config["mixup_alpha"], num_classes=100),
                       v2.CutMix(alpha=self.config["cutmix_alpha"], num_classes=100)):
            with self.subTest(policy=type(policy).__name__):
                torch.manual_seed(7)
                mixed, soft_labels = policy(images.clone(), labels)
                self.assertEqual(mixed.shape, images.shape)
                self.assertEqual(soft_labels.shape, (4, 100))
                torch.testing.assert_close(soft_labels.sum(dim=1), torch.ones(4))
                self.assertTrue((soft_labels >= 0).all())
                self.assertTrue((soft_labels <= 1).all())
                logits = torch.zeros(4, 100, requires_grad=True)
                F.cross_entropy(logits, soft_labels).backward()
                torch.testing.assert_close(logits.grad, (0.01 - soft_labels) / 4)

    def test_evaluation_uses_all_images_with_an_uneven_last_batch(self):
        labels = torch.cat([torch.arange(100), torch.tensor([0, 0, 0])])
        logits = torch.zeros(103, 100)
        logits[torch.arange(100), torch.arange(100)] = 2
        logits[100:, 1] = 2
        loader = DataLoader(TensorDataset(logits, labels), batch_size=16)
        model = torch.nn.Identity()
        scores, targets, predictions = evaluate(model, loader, device="cpu")
        self.assertFalse(model.training)
        self.assertEqual(targets, labels.tolist())
        self.assertEqual(predictions, logits.argmax(dim=1).tolist())
        self.assertAlmostEqual(scores["accuracy"], 100 * 100 / 103)
        self.assertAlmostEqual(scores["macro_f1"], 98.8)
        self.assertAlmostEqual(scores["balanced_accuracy"], 99.25)
        self.assertAlmostEqual(scores["validation_loss"], F.cross_entropy(logits, labels).item(), places=5)


if __name__ == "__main__":
    unittest.main()
