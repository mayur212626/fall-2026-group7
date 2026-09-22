# CIFAR-100 real-data baselines

We trained ResNet-50, ConvNeXt-Tiny, ViT-B/16, and Swin-T from scratch using seeds 0, 1, 2, 3, and 4.

Each run used 45,000 training images, 5,000 validation images, batch size 64, and 4,218 optimizer steps. The data split was fixed at seed 42. The official test set was not used.

The summary reports validation metrics at the final step, averaged across five seeds with sample standard deviation. Metrics use a 0–100 scale.

Each run includes its configuration and metric history. The runner's result.json records its best validation checkpoint; summary.json uses the final step from history.jsonl.

These are six-epoch, fixed-budget baselines. Convergence has not been established. Optimizer settings differ by model and are recorded in the configurations.

Datasets and model checkpoints are stored separately.
