# CIFAR-100 splits and metrics

Run these commands from the project folder with the `synthaug-bench` environment active:

```bash
python -m unittest discover -s tests -v
python scripts/prepare_cifar100.py
```

The split script uses the existing download under `data/cifar100` and the settings in `configs/cifar100_fewshot.json`. It does not download another copy or evaluate a model. The tests use fixture labels and small prediction examples; running them does not validate the actual dataset files.

The script saves `configs/splits/cifar100_fewshot.json`. Its indices refer to the official training split as loaded by Torchvision. It records the class names, label hash, NumPy version, sampling algorithm, and split seed. An existing manifest with different contents stops the script so a recorded split is not silently replaced. The small manifest can be committed; the image data stays outside Git.

With the current configuration, validation contains 5,000 images. The four nested training subsets contain 500, 1,000, 2,000, and 5,000 images. Every class contributes the configured number of images, and training and validation do not overlap. The official test split stays separate. The five training seeds do not create five different data splits.

`scripts/metrics.py` calculates top-1 accuracy, macro-F1, balanced accuracy, per-class recall, class counts, and a confusion matrix. Scores use a 0-100 scale. Matrix rows are true labels and columns are predictions. The function requires integer class IDs from zero to the class count minus one and refuses an evaluation set missing a class. Collect predictions for the whole evaluation split before calling it, rather than averaging batch-level macro-F1 scores.

The checks cover the sampling counts, nested subsets, train/validation separation, repeated and changed sampling seeds, a hand-calculated imbalanced example, perfect predictions, a class with no correct predictions, invalid inputs, and small integer label types.

The training runner will add validation cross-entropy, checkpoint selection, runtime, and GPU memory measurements. Paired five-seed summaries will follow actual training. No classifier score or training result is produced by this step.

References: [Torchvision CIFAR100](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html), [macro-F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).
