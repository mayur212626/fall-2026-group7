"""Ensembles of a classifier trained with real images and one trained with generated images.

Two steps, because the checkpoints of the real-only and the synthetic-data runs
live on different machines:

    export   For every completed run below ``--root``, rebuild the model from
             ``best.pt`` and save its softmax probabilities on the validation set
             (and on the test set, for runs already scored on it) to ``probs.npz``
             in the run folder. The small ``probs.npz`` files are what the
             machines exchange; checkpoints never move.
    combine  Average the probabilities of two runs with equal weights, nothing
             tuned, and score the average.

Every ensemble pairs two runs of the same model, initialization and budget.
An ensemble of two different conditions pairs equal training seeds; an ensemble
of one condition with itself (the control) pairs seed s with the next seed,
since any two models averaged beat one. The main comparison is therefore

    (real-only + SD) ensemble  minus  (real-only + real-only) ensemble

and likewise with RandAugment, paired by the seed s of the first member, with
the verdict rule of the other stages.

    python -m src.component.ensemble export --root runs/stage1-pretrained --splits validation test
    python -m src.component.ensemble combine --split validation
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.component.analyze_results import MODEL_NAMES, VERDICT_MARKS, _budget_key, _tex, write_csv

ROOTS = ["runs/stage1-{init}", "runs/stage2-convnext-{init}", "runs/augment-{init}"]
# ensemble name -> (first member, second member); identical members pair seed s with the next seed
ENSEMBLES = {
    "real+real": ("real_only", "real_only"),
    "RA+RA": ("randaugment", "randaugment"),
    "real+SD": ("real_only", "sd_prompt"),
    "RA+SD": ("randaugment", "sd_prompt"),
    "real+SDonly": ("real_only", "sd_prompt_only"),
    "RA+SDonly": ("randaugment", "sd_prompt_only"),
    "RA+SD+RA": ("randaugment", "sd_prompt_randaugment"),
    "SD+RA x2": ("sd_prompt_randaugment", "sd_prompt_randaugment"),
}
# (treatment ensemble, control ensemble or single condition, label) reported in the tables
COMPARISONS = [
    ("real+SD", "real+real", "(real+SD)$-$(real+real)"),
    ("real+SDonly", "real+real", "(real+SDonly)$-$(real+real)"),
    ("RA+SD", "RA+RA", "(RA+SD)$-$(RA+RA)"),
    ("RA+SDonly", "RA+RA", "(RA+SDonly)$-$(RA+RA)"),
    ("RA+SD+RA", "RA+RA", "(RA+SD+RA)$-$(RA+RA)"),
    ("RA+SD+RA", "sd_prompt_randaugment", "(RA+SD+RA)$-$SD+RA"),
]
CONDITIONS = sorted({c for pair in ENSEMBLES.values() for c in pair})


# ---------------------------------------------------------------- export

def export_run(run_dir: Path, sets: dict, device: "torch.device") -> Path:  # noqa: F821
    """Save the softmax probabilities of a run's best checkpoint on each split in ``sets``.

    ``sets`` maps a split name to ``(CifarImages, labels)``. Probabilities are stored as
    float16 in the evaluation order (image IDs ascending within the split's index list).
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from src.component.data import evaluation_batches, to_model_input
    from src.component.models import build_model

    settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"]
    model = build_model(settings["model"], pretrained=False, num_classes=settings["num_classes"])
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    model.to(device).eval()
    out = {"checkpoint_step": np.array(best["step"])}
    for split, (dataset, labels) in sets.items():
        loader = DataLoader(dataset, batch_sampler=evaluation_batches(len(dataset), 256), num_workers=4,
                            pin_memory=device.type == "cuda")
        probs, ids = [], []
        with torch.no_grad():
            for images, _, image_ids in loader:
                inputs = to_model_input(images.to(device, non_blocking=True), settings["image_size"])
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(inputs)
                probs.append(F.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float16))
                ids += image_ids.tolist()
        out[f"{split}_probs"] = np.concatenate(probs)
        out[f"{split}_ids"] = np.asarray(ids, dtype=np.int32)
        out[f"{split}_labels"] = np.asarray(labels, dtype=np.int16)
    path = run_dir / "probs.npz"
    tmp = run_dir / "probs.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.replace(path)
    return path


def export(root: Path, splits: list[str], conditions: list[str], data_root: Path, manifest: Path) -> None:
    """Export ``probs.npz`` for every completed run of ``conditions`` below ``root`` that lacks one.

    A test split is exported only for runs that already have ``test_result.json``,
    so this never scores a run on the test set before the protocol allows it.
    """
    import torch

    from src.component.data import CifarImages, load_cifar100, load_manifest

    images, labels = load_cifar100(data_root, train=True)
    validation_ids = load_manifest(manifest)["validation_indices"]
    sets = {"validation": (CifarImages(images, labels, validation_ids, randaugment=False),
                           [labels[i] for i in validation_ids])}
    if "test" in splits:
        test_images, test_labels = load_cifar100(data_root, train=False)
        sets["test"] = (CifarImages(test_images, test_labels, list(range(len(test_labels))), randaugment=False),
                        list(test_labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for result_path in sorted(Path(root).rglob("result.json")):
        run_dir = result_path.parent
        settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"]
        if settings["condition"] not in conditions or settings.get("synthetic", {}).get("ratio", 1) != 1:
            continue
        if (run_dir / "probs.npz").exists():
            logging.info("skip %s (probs.npz exists)", run_dir.name)
            continue
        if not (run_dir / "best.pt").exists():
            logging.warning("skip %s: no best.pt on this machine", run_dir.name)
            continue
        wanted = {s: v for s, v in sets.items()
                  if s == "validation" or (run_dir / "test_result.json").exists()}
        export_run(run_dir, wanted, device)
        logging.info("%s: exported %s", run_dir.name, ", ".join(wanted))


# ---------------------------------------------------------------- combine

def load_probs(roots: list[str], split: str) -> dict[tuple, dict]:
    """Probabilities of every exported run, keyed by (init, model, budget, condition, seed)."""
    runs = {}
    for init in ("pretrained", "scratch"):
        for pattern in roots:
            root = Path(pattern.format(init=init))
            if not root.exists():
                continue
            for path in sorted(root.rglob("probs.npz")):
                settings = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))["settings"]
                data = np.load(path)
                if f"{split}_probs" not in data:
                    continue
                key = (settings["init"], settings["model"], settings["budget"], settings["condition"],
                       settings["seed"])
                runs[key] = {"probs": data[f"{split}_probs"].astype(np.float32),
                             "ids": data[f"{split}_ids"], "labels": data[f"{split}_labels"]}
    return runs


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy in percent."""
    return float((probs.argmax(axis=1) == labels).mean() * 100)


def ensemble_scores(runs: dict[tuple, dict]) -> list[dict]:
    """One row per (ensemble or single condition, init, model, budget, seed) with its accuracy.

    Raises:
        ValueError: If two members were scored on different images or in a different order.
    """
    rows = []
    cells = sorted({k[:3] for k in runs})
    for init, model, budget in cells:
        by_condition: dict[str, dict[int, dict]] = {}
        for (i, m, b, condition, seed), run in runs.items():
            if (i, m, b) == (init, model, budget):
                by_condition.setdefault(condition, {})[seed] = run
        for condition, seeds in by_condition.items():
            for seed, run in seeds.items():
                rows.append({"init": init, "model": model, "budget": budget, "condition": condition, "seed": seed,
                             "partner_seed": None, "accuracy": accuracy(run["probs"], run["labels"])})
        for name, (first, second) in ENSEMBLES.items():
            a, b = by_condition.get(first, {}), by_condition.get(second, {})
            seeds = sorted(a)
            for k, seed in enumerate(seeds):
                partner = (seeds[(k + 1) % len(seeds)] if len(seeds) > 1 else None) if first == second else seed
                if partner is None or partner not in b:
                    continue
                x, y = a[seed], b[partner]
                if not np.array_equal(x["ids"], y["ids"]):
                    raise ValueError(f"{init} {model} b{budget} {name}: members scored on different images")
                rows.append({"init": init, "model": model, "budget": budget, "condition": name, "seed": seed,
                             "partner_seed": partner,
                             "accuracy": accuracy((x["probs"] + y["probs"]) / 2, x["labels"])})
    return rows


def comparisons(rows: list[dict]) -> list[dict]:
    """Paired change by seed of each comparison in ``COMPARISONS``, with the all-seeds-agree verdict."""
    from src.component.analyze_results import paired_changes

    out = []
    for treatment, control, label in COMPARISONS:
        for c in paired_changes(rows, treatment, control, "accuracy"):
            out.append({"treatment": treatment, "control": control, "label": label, **c})
    return out


def table(changes: list[dict], init: str, split: str) -> str:
    """Booktabs table of the ensemble comparisons for one initialization."""
    labels = [label for _, _, label in COMPARISONS]
    cell = {(c["model"], c["budget"], c["label"]): c for c in changes if c["init"] == init}
    seeds = sorted({len(c["seeds"]) for c in cell.values()})
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{Ensembles, {init} initialization: change in {split} top-1 accuracy (percentage points) of an "
        r"equal-weight average of two classifiers' softmax outputs. Each ensemble of a real-data and a "
        r"generated-data classifier is compared with an ensemble of two real-data classifiers of different seeds, "
        rf"paired by seed, mean $\pm$ sample SD over {' or '.join(map(str, seeds))} seeds. SDonly: trained on "
        r"SD class-prompt images alone; SD+RA: real and SD images with RandAugment. $\uparrow$ gain and "
        r"$\downarrow$ loss when every seed agrees in sign, $\sim$ inconclusive otherwise.}",
        rf"\label{{tab:ensemble-{init}{'-test' if split == 'test' else ''}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(labels) + "}",
        r"\toprule",
        "Model & Images/class & " + " & ".join(labels) + r" \\",
    ]
    for model in MODEL_NAMES:
        budgets = sorted({b for m, b, _ in cell if m == model}, key=_budget_key)
        if not budgets:
            continue
        lines.append(r"\midrule")
        for budget in budgets:
            cells = [MODEL_NAMES[model], budget]
            for label in labels:
                c = cell.get((model, budget, label))
                cells.append(f"{_tex(c['mean'], c['sd'], '+')} {VERDICT_MARKS[c['verdict']]}" if c else "--")
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def combine(roots: list[str], split: str, tables: Path) -> list[dict]:
    """Score all ensembles of one split and write the CSVs and one table per initialization."""
    rows = ensemble_scores(load_probs(roots, split))
    if not rows:
        raise SystemExit(f"no probs.npz with the {split} split below {roots}; run the export step first")
    changes = comparisons(rows)
    suffix = "_test" if split == "test" else ""
    print(f"wrote {write_csv(rows, tables / f'ensemble_accuracy{suffix}.csv')}")
    if changes:
        print(f"wrote {write_csv(changes, tables / f'ensemble_paired{suffix}.csv')}")
    for init in ("pretrained", "scratch"):
        if any(c["init"] == init for c in changes):
            path = tables / f"ensemble_{init}{suffix}.tex"
            path.write_text(table(changes, init, split), encoding="utf-8")
            print(f"wrote {path}")
    print(f"\nensembles ({split}): paired change (pp), mean over seeds, verdict")
    for c in changes:
        print(f"  {c['init']:<10} {MODEL_NAMES.get(c['model'], c['model']):<10} b{c['budget']:<4} "
              f"{c['label']:<26} {c['mean']:+6.2f}  ({len(c['seeds'])} seeds, {c['verdict']})")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="save probs.npz for the completed runs below --root")
    ex.add_argument("--root", type=Path, required=True)
    ex.add_argument("--splits", nargs="+", default=["validation"], choices=["validation", "test"])
    ex.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ex.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    ex.add_argument("--manifest", type=Path, default=Path("src/component/configs/splits/cifar100.json"))
    co = sub.add_parser("combine", help="score the ensembles from the exported probs.npz files")
    co.add_argument("--split", default="validation", choices=["validation", "test"])
    co.add_argument("--roots", nargs="+", default=ROOTS, help="run folders, with {init} for the initialization")
    co.add_argument("--tables", type=Path, default=Path("reports/Latex_report/tables"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "export":
        export(args.root, args.splits, args.conditions, args.data_root, args.manifest)
    else:
        args.tables.mkdir(parents=True, exist_ok=True)
        combine(args.roots, args.split, args.tables)


if __name__ == "__main__":
    main()
