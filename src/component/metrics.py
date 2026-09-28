"""Classification metrics shared by every run.

All scores are on a 0-100 scale. The class list is fixed by ``num_classes``,
so every run reports results in the same class order.
"""

from typing import Sequence

import numpy as np


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> dict:
    """Compute accuracy, macro-F1, balanced accuracy and per-class results.

    Macro-F1 is the unweighted mean of the per-class F1 scores over all
    ``num_classes`` classes; a class whose F1 is undefined (no true and no
    predicted images, or zero precision and recall) contributes 0. Balanced
    accuracy is the mean recall over the classes present in ``y_true``.
    Classes without evaluation images are listed in ``missing_classes`` and
    have a recall of ``None``.

    Args:
        y_true: True class of each evaluation image.
        y_pred: Predicted class of each evaluation image.
        num_classes: Number of classes in the fixed class list.

    Returns:
        Dictionary with ``accuracy``, ``macro_f1``, ``balanced_accuracy``,
        ``per_class_recall``, ``class_counts``, ``missing_classes`` and
        ``confusion_matrix`` (rows are true classes, columns predictions).

    Raises:
        ValueError: If the inputs are empty, differ in length, or contain a
            label outside ``range(num_classes)``.
    """
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.size == 0 or true.shape != pred.shape:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    for labels in (true, pred):
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"labels must be in range(0, {num_classes})")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (true, pred), 1)
    correct = np.diag(confusion)
    true_counts = confusion.sum(axis=1)
    pred_counts = confusion.sum(axis=0)
    present = true_counts > 0

    recall = np.divide(correct, true_counts, out=np.zeros(num_classes), where=present)
    f1_denominator = true_counts + pred_counts
    f1 = np.divide(2 * correct, f1_denominator, out=np.zeros(num_classes), where=f1_denominator > 0)

    return {
        "accuracy": 100 * float(correct.sum()) / true.size,
        "macro_f1": 100 * float(f1.mean()),
        "balanced_accuracy": 100 * float(recall[present].mean()),
        "per_class_recall": [100 * float(r) if p else None for r, p in zip(recall, present)],
        "class_counts": true_counts.tolist(),
        "missing_classes": np.flatnonzero(~present).tolist(),
        "confusion_matrix": confusion.tolist(),
    }
