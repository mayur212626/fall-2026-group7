# CIFAR-100 data splits

Every run reads its images from one saved manifest, so all conditions, models and seeds use exactly the same validation and training images.

## Split design

| Split | Images |
|---|---|
| Validation | 50 per class (5,000), taken from the official training set |
| Training budgets | 5, 10, 20 and 50 per class, and `full` (all 45,000 non-validation images) |
| Test | Official 10,000-image test set, not in the manifest |

Each class's indices are shuffled once with split seed 42 (NumPy PCG64). The first 50 go to validation. A budget of `k` images per class takes the next `k`, so every budget contains the smaller ones.

The manifest also stores a SHA-256 fingerprint of the training labels, so a run can check it reads the same dataset.

## Build the manifest

From the repository root, with CIFAR-100 already downloaded to `data/`:

```bash
python -m src.component.cifar_splits --data-root data --output src/component/configs/splits/cifar100.json
```

Expected output:

```
validation: 5000, training: {'5': 500, '10': 1000, '20': 2000, '50': 5000, 'full': 45000}
```

## Tests

```bash
python -m unittest discover -s src/tests -v
```

The tests use small synthetic labels and do not need the dataset.
