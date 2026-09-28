"""One-time evaluation of completed runs on the official CIFAR-100 test set.

The test set is scored only after the training and selection protocol is
fixed, so the command requires ``--confirm-protocol-frozen``. Each completed
run (a directory with ``result.json`` and ``best.pt``) is scored once with
its best validation checkpoint; the result is saved as ``test_result.json``
and is never overwritten. Runs that already have one are skipped.

    python -m src.component.evaluate_test --root runs/stage1-pretrained --confirm-protocol-frozen
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.component.cifar_splits import labels_sha256
from src.component.data import CifarImages, load_cifar100
from src.component.models import build_model
from src.component.train import evaluate


def evaluate_run(run_dir: Path, test_set: CifarImages, device: torch.device, labels_hash: str) -> dict:
    """Score the best checkpoint of a completed run on ``test_set`` and save ``test_result.json``.

    The model is rebuilt from the run's settings without downloading pretrained
    weights, then loaded from ``best.pt``. Scoring uses the same ``evaluate``
    function as validation during training.

    Raises:
        FileNotFoundError: If the run has no ``result.json`` (not complete).
        FileExistsError: If the run already has a ``test_result.json``.
    """
    run_dir = Path(run_dir)
    out_path = run_dir / "test_result.json"
    if not (run_dir / "result.json").exists():
        raise FileNotFoundError(f"{run_dir} is not a completed run")
    if out_path.exists():
        raise FileExistsError(f"{run_dir} was already evaluated on the test set")

    settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"]
    model = build_model(settings["model"], pretrained=False, num_classes=settings["num_classes"])
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    metrics, predictions = evaluate(model.to(device), test_set, settings, device)

    result = {
        "run": run_dir.name,
        "checkpoint_step": best["step"],
        "evaluated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_labels_sha256": labels_hash,
        "test": metrics,
        "test_predictions": predictions,
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Score every completed, not yet evaluated run below ``--root`` on the test set."""
    parser = argparse.ArgumentParser(description="Evaluate completed runs on the CIFAR-100 test set once.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--confirm-protocol-frozen", action="store_true",
                        help="required: confirms the training and selection protocol is final")
    args = parser.parse_args()
    if not args.confirm_protocol_frozen:
        parser.error("the test set is used once, after the protocol is fixed; pass --confirm-protocol-frozen")

    images, labels = load_cifar100(args.data_root, train=False)
    test_set = CifarImages(images, labels, list(range(len(labels))), randaugment=False)
    labels_hash = labels_sha256(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for result_path in sorted(args.root.rglob("result.json")):
        run_dir = result_path.parent
        if (run_dir / "test_result.json").exists():
            print(f"skip {run_dir.name} (already evaluated)")
            continue
        out = evaluate_run(run_dir, test_set, device, labels_hash)
        print(f"{run_dir.name}: test accuracy {out['test']['accuracy']:.2f}, macro-F1 {out['test']['macro_f1']:.2f}")


if __name__ == "__main__":
    main()
