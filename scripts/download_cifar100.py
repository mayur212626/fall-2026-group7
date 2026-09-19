from collections import Counter
from pathlib import Path

from torchvision.datasets import CIFAR100

root = Path(__file__).resolve().parents[1] / "data" / "cifar100"

for split, training, per_class in (("train", True, 500), ("test", False, 100)):
    dataset = CIFAR100(root=root, train=training, download=True)
    assert len(dataset.classes) == 100, f"{split}: unexpected class count"
    assert dataset.data.shape == (100 * per_class, 32, 32, 3), f"{split}: unexpected image dimensions"
    assert Counter(dataset.targets) == {i: per_class for i in range(100)}, f"{split}: unexpected label counts"
    print(f"{split}: {len(dataset)} images, 100 classes, {per_class} images per class")
