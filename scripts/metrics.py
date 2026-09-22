import numpy as np


def classification_metrics(targets, predictions, num_classes):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.ndim != 1 or targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("Targets and predictions must be nonempty matching one-dimensional arrays.")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer.")
    for labels in (targets, predictions):
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must be integers.")
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError("Labels must be between 0 and num_classes - 1.")

    targets = targets.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    matrix = np.bincount(
        num_classes * targets + predictions, minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    counts = matrix.sum(axis=1)
    missing = np.flatnonzero(counts == 0).tolist()
    if missing:
        raise ValueError(f"Evaluation set is missing classes: {missing}")
    correct = matrix.diagonal()
    recall = correct / counts
    f1 = 2 * correct / (counts + matrix.sum(axis=0))

    return {
        "accuracy": float(100 * correct.sum() / targets.size),
        "macro_f1": float(100 * f1.mean()),
        "balanced_accuracy": float(100 * recall.mean()),
        "per_class_recall": (100 * recall).tolist(),
        "class_counts": counts.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
