"""Error analysis of Stage 1 from the stored validation predictions.

Every run keeps its best-checkpoint prediction for each validation image in
``result.json``, so this analysis needs no retraining and never touches the
test split. For one data budget and a pair of conditions (a treatment and its
control, paired by training seed) it writes:

- the change in recall of each class and each CIFAR-100 superclass, as the
  mean over seeds of treatment minus control;
- the confusions (true class, predicted class) whose error count changed the
  most, as the mean over seeds;
- for every model and budget, the paired accuracy change with a 95%
  percentile bootstrap interval over validation images. This interval covers
  image-sampling noise only; the spread over seeds is reported separately as
  the sample SD of the per-seed changes;
- a two-sided sign test of the mean changes across all models and budgets.

The class-to-superclass map comes from the CIFAR-100 dataset files; write it
once on the machine that has the data:

    python -m src.component.error_analysis superclasses --data-root data \
        --output src/component/configs/cifar100_superclasses.csv

then analyze one initialization:

    python -m src.component.error_analysis analyze --root runs/stage1-pretrained \
        --superclasses src/component/configs/cifar100_superclasses.csv \
        --tables reports/Latex_report/tables --figures reports/Latex_report/fig
"""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np

from src.component.analyze_results import (
    CONDITION_NAMES,
    MODEL_NAMES,
    _budget_key,
    paired_changes,
    write_csv,
)

NUM_CLASSES = 100
NUM_SUPERCLASSES = 20
GAIN_COLOR = "#2a78d6"  # diverging pair, validated: blue for a gain, red for a loss
LOSS_COLOR = "#e34948"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
# (treatment, control) pairs analyzed per class, superclass and confusion.
ERROR_COMPARISONS = [("sd_prompt", "real_only"), ("sd_prompt", "randaugment"),
                     ("sd_prompt_randaugment", "randaugment")]
# (treatment, control) pairs with bootstrap intervals and a sign test.
CI_COMPARISONS = [("randaugment", "real_only"), ("sd_prompt", "real_only"), ("sd_lora", "real_only"),
                  ("sd_prompt_randaugment", "real_only"), ("sd_prompt", "randaugment"),
                  ("sd_lora", "randaugment"), ("sd_prompt_randaugment", "randaugment")]
SHORT_NAMES = {"real_only": "real", "randaugment": "RA", "sd_prompt": "SD", "sd_lora": "LoRA",
               "sd_prompt_randaugment": "SD+RA"}


def load_predictions(root: Path) -> list[dict]:
    """Return one entry per completed run below ``root``: its settings and predictions ordered by image id."""
    runs = []
    for result_path in sorted(Path(root).rglob("result.json")):
        settings = json.loads((result_path.parent / "config.json").read_text(encoding="utf-8"))["settings"]
        predictions = sorted(json.loads(result_path.read_text(encoding="utf-8"))["validation_predictions"],
                             key=lambda p: p["image_id"])
        runs.append({
            **{k: settings[k] for k in ("init", "model", "condition", "budget", "seed")},
            "image_ids": np.array([p["image_id"] for p in predictions]),
            "true": np.array([p["true"] for p in predictions]),
            "pred": np.array([p["pred"] for p in predictions]),
        })
    return runs


def per_class_recall(true: np.ndarray, pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Recall (%) of every class."""
    correct = np.bincount(true[pred == true], minlength=num_classes)
    return 100.0 * correct / np.bincount(true, minlength=num_classes)


def paired_runs(runs: list[dict], init: str, model: str, budget: str, treatment: str,
                control: str) -> list[tuple[dict, dict]]:
    """(treatment run, control run) for every seed both conditions have; both must score the same images."""
    cell = {(r["condition"], r["seed"]): r for r in runs
            if (r["init"], r["model"], r["budget"]) == (init, model, budget)}
    seeds = sorted({s for c, s in cell if c == treatment} & {s for c, s in cell if c == control})
    pairs = [(cell[(treatment, s)], cell[(control, s)]) for s in seeds]
    for t, c in pairs:
        if not np.array_equal(t["image_ids"], c["image_ids"]) or not np.array_equal(t["true"], c["true"]):
            raise ValueError(f"{init} {model} b{budget} seed {t['seed']}: runs scored different validation images")
    return pairs


def class_recall_change(pairs: list[tuple[dict, dict]], num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Mean over seeds of treatment minus control recall (percentage points) per class."""
    return np.mean([per_class_recall(t["true"], t["pred"], num_classes) - per_class_recall(c["true"], c["pred"], num_classes)
                    for t, c in pairs], axis=0)


def superclass_change(class_change: np.ndarray, coarse_of: list[int],
                      num_superclasses: int = NUM_SUPERCLASSES) -> np.ndarray:
    """Mean class change within each superclass."""
    coarse = np.asarray(coarse_of)
    return np.array([class_change[coarse == s].mean() for s in range(num_superclasses)])


def confusion_changes(pairs: list[tuple[dict, dict]]) -> list[dict]:
    """Mean errors per seed for each (true, predicted) pair in either condition, and their change."""
    def errors(runs: list[dict]) -> dict[tuple[int, int], float]:
        counts: dict[tuple[int, int], float] = {}
        for r in runs:
            wrong = r["pred"] != r["true"]
            for key in zip(r["true"][wrong].tolist(), r["pred"][wrong].tolist(), strict=True):
                counts[key] = counts.get(key, 0) + 1 / len(runs)
        return counts

    treated, controls = errors([t for t, _ in pairs]), errors([c for _, c in pairs])
    return [{"true": k[0], "pred": k[1], "treatment": treated.get(k, 0.0), "control": controls.get(k, 0.0),
             "change": treated.get(k, 0.0) - controls.get(k, 0.0)} for k in sorted(set(treated) | set(controls))]


def bootstrap_ci(pairs: list[tuple[dict, dict]], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of the seed-mean accuracy change, resampling validation images.

    The same resampled images are used for every seed and both conditions, so
    the pairing is kept.
    """
    per_image = np.mean([(t["pred"] == t["true"]).astype(float) - (c["pred"] == c["true"]).astype(float)
                         for t, c in pairs], axis=0)
    rng = np.random.default_rng(seed)
    means = np.concatenate([per_image[rng.integers(0, len(per_image), (n, len(per_image)))].mean(axis=1)
                            for n in [1000] * (resamples // 1000) + [resamples % 1000] if n])
    low, high = np.quantile(100 * means, [(1 - level) / 2, (1 + level) / 2])
    return round(float(low), 10), round(float(high), 10)


def sign_test(changes: list[float]) -> float:
    """Two-sided exact sign test p-value of the changes; zero changes are left out."""
    signs = [c > 0 for c in changes if c != 0]
    n, k = len(signs), min(sum(signs), len(signs) - sum(signs))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def write_superclasses(data_root: Path, output: Path) -> Path:
    """Write the class-to-superclass map of CIFAR-100, read from the dataset files below ``data_root``."""
    folder = Path(data_root) / "cifar-100-python"
    meta = pickle.loads((folder / "meta").read_bytes(), encoding="latin1")
    train = pickle.loads((folder / "train").read_bytes(), encoding="latin1")
    coarse_of: dict[int, int] = {}
    for fine, coarse in zip(train["fine_labels"], train["coarse_labels"], strict=True):
        if coarse_of.setdefault(fine, coarse) != coarse:
            raise ValueError(f"class {fine} belongs to superclasses {coarse_of[fine]} and {coarse}")
    rows = [{"class": f, "class_name": meta["fine_label_names"][f], "superclass": coarse_of[f],
             "superclass_name": meta["coarse_label_names"][coarse_of[f]]} for f in sorted(coarse_of)]
    return write_csv(rows, output)


def read_superclasses(path: Path) -> dict:
    """Class names, superclass of each class and superclass names from ``write_superclasses`` output."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["class"]))
    names = {int(r["superclass"]): r["superclass_name"] for r in rows}
    return {"class_names": [r["class_name"] for r in rows], "coarse_of": [int(r["superclass"]) for r in rows],
            "superclass_names": [names[s] for s in sorted(names)]}


def plot_superclass_change(changes: dict[str, np.ndarray], names: list[str], out_stem: Path,
                           xlabel: str) -> list[Path]:
    """Diverging horizontal bars of the change per superclass, one panel per model.

    Superclasses are sorted by the mean change over the models, largest on
    top; blue bars are gains and red bars losses. Saves SVG and PDF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    order = np.argsort(np.mean(list(changes.values()), axis=0), kind="stable")
    limit = max(abs(v).max() for v in changes.values()) * 1.1
    fig, axes = plt.subplots(1, len(changes), figsize=(3.0 * len(changes) + 1.6, 0.2 * len(names) + 1.0),
                             sharey=True)
    for ax, (model, change) in zip(np.atleast_1d(axes), changes.items(), strict=True):
        values = change[order]
        ax.barh(range(len(values)), values, height=0.7, color=[GAIN_COLOR if v > 0 else LOSS_COLOR for v in values])
        ax.axvline(0, color="#8a8a86", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_title(MODEL_NAMES[model])
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", color="#e0e0e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax = np.atleast_1d(axes)[0]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i].replace("_", " ") for i in order])
    fig.tight_layout()

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_stem.with_suffix(".svg"), out_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def interval_rows(runs: list[dict], init: str) -> list[dict]:
    """Paired accuracy change, seed SD, verdict and bootstrap interval of every CI comparison and cell."""
    accuracy = [{**{k: r[k] for k in ("init", "model", "condition", "budget", "seed")},
                 "accuracy": 100.0 * float(np.mean(r["pred"] == r["true"]))} for r in runs if r["init"] == init]
    rows = []
    for treatment, control in CI_COMPARISONS:
        for c in paired_changes(accuracy, treatment, control, "accuracy"):
            low, high = bootstrap_ci(paired_runs(runs, init, c["model"], c["budget"], treatment, control))
            rows.append({"init": init, "model": c["model"], "budget": c["budget"], "treatment": treatment,
                         "control": control, "seeds": c["seeds"], "changes": c["changes"], "mean": c["mean"],
                         "seed_sd": c["sd"], "ci_low": low, "ci_high": high, "verdict": c["verdict"],
                         "resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED})
    return rows


def sign_test_rows(intervals: list[dict]) -> list[dict]:
    """Sign test of the mean changes over all models and budgets, per comparison."""
    rows = []
    for treatment, control in CI_COMPARISONS:
        means = [r["mean"] for r in intervals if (r["treatment"], r["control"]) == (treatment, control)]
        if means:
            rows.append({"init": intervals[0]["init"], "treatment": treatment, "control": control, "cells": len(means),
                         "gains": sum(m > 0 for m in means), "losses": sum(m < 0 for m in means),
                         "p_value": sign_test(means)})
    return rows


def interval_table(intervals: list[dict], tests: list[dict], init: str) -> str:
    """Booktabs table of the paired change with its 95% bootstrap interval, for the main comparisons."""
    shown = [("sd_prompt", "real_only"), ("sd_prompt", "randaugment"), ("sd_prompt_randaugment", "randaugment")]
    cells = {(r["model"], r["budget"], r["treatment"], r["control"]): r for r in intervals}
    p_values = {(t["treatment"], t["control"]): t for t in tests}
    header = [f"{SHORT_NAMES[t]}$-${SHORT_NAMES[c]} (pp) $\\uparrow$" for t, c in shown]
    resamples = f"{BOOTSTRAP_RESAMPLES:,}".replace(",", "{,}")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Stage 1, {init} initialization: change in validation top-1 accuracy (percentage points), "
        r"mean over training seeds of the paired change, with its 95\% percentile bootstrap interval over the "
        rf"5{{,}}000 validation images ({resamples} resamples, the same images for every seed and "
        r"both conditions). The interval covers image-sampling noise only; the seed SD is in the paired tables. "
        r"The last row is a two-sided sign test of the mean changes over all cells (gains/cells, $p$). "
        r"SD: pretrained Stable Diffusion with class prompts; RA: RandAugment.}",
        rf"\label{{tab:stage1-{init}-bootstrap}}",
        r"\begin{tabular}{ll" + "r" * len(shown) + "}",
        r"\toprule",
        "Model & Images/class & " + " & ".join(header) + r" \\",
    ]
    for model in MODEL_NAMES:
        budgets = sorted({b for m, b, _, _ in cells if m == model}, key=_budget_key)
        if not budgets:
            continue
        lines.append(r"\midrule")
        for budget in budgets:
            row = [MODEL_NAMES[model], budget]
            for t, c in shown:
                r = cells.get((model, budget, t, c))
                row.append(f"{r['mean']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]" if r else "--")
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    signs = []
    for t, c in shown:
        p = p_values.get((t, c))
        signs.append(f"{p['gains']}/{p['cells']}, $p={p['p_value']:.3f}$" if p else "--")
    lines += [r"\multicolumn{2}{l}{Sign test} & " + " & ".join(signs) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def analyze(root: Path, superclasses_path: Path, tables: Path, figures: Path, budget: str, top: int) -> None:
    """Write the error-analysis tables and figures of every initialization below ``root``."""
    runs = load_predictions(root)
    classes = read_superclasses(superclasses_path)
    for init in sorted({r["init"] for r in runs}):
        intervals = interval_rows(runs, init)
        tests = sign_test_rows(intervals)
        write_csv(intervals, tables / f"stage1_{init}_bootstrap.csv")
        write_csv(tests, tables / f"stage1_{init}_sign_test.csv")
        (tables / f"stage1_{init}_bootstrap.tex").write_text(interval_table(intervals, tests, init), encoding="utf-8")
        for t in tests:
            print(f"{init} {t['treatment']} - {t['control']}: {t['gains']}/{t['cells']} gains, sign test p={t['p_value']:.4f}")

        for treatment, control in ERROR_COMPARISONS:
            name = f"stage1_{init}_b{budget}_{treatment}_vs_{control}"
            class_rows, superclass_rows, confusion_rows, plotted = [], [], [], {}
            for model in MODEL_NAMES:
                pairs = paired_runs(runs, init, model, budget, treatment, control)
                if not pairs:
                    continue
                seeds = [t["seed"] for t, _ in pairs]
                change = class_recall_change(pairs)
                plotted[model] = superclass_change(change, classes["coarse_of"])
                class_rows += [{"model": model, "class": k, "class_name": classes["class_names"][k],
                                "superclass_name": classes["superclass_names"][classes["coarse_of"][k]],
                                "recall_change": float(change[k]), "seeds": seeds} for k in np.argsort(change)]
                superclass_rows += [{"model": model, "superclass": s, "superclass_name": classes["superclass_names"][s],
                                     "recall_change": float(v), "seeds": seeds} for s, v in enumerate(plotted[model])]
                confusions = sorted(confusion_changes(pairs), key=lambda c: c["change"])
                confusion_rows += [{"model": model, "effect": effect, "true": classes["class_names"][c["true"]],
                                    "pred": classes["class_names"][c["pred"]], "treatment_errors": c["treatment"],
                                    "control_errors": c["control"], "change": c["change"], "seeds": seeds}
                                   for effect, part in (("fixed", confusions[:top]), ("created", confusions[::-1][:top]))
                                   for c in part]
            if not plotted:
                continue
            write_csv(class_rows, tables / f"{name}_classes.csv")
            write_csv(superclass_rows, tables / f"{name}_superclasses.csv")
            write_csv(confusion_rows, tables / f"{name}_confusions.csv")
            label = f"Recall change, {SHORT_NAMES[treatment]} $-$ {SHORT_NAMES[control]} (pp)"
            plot_superclass_change(plotted, classes["superclass_names"], figures / f"{name}_superclasses", label)
            print(f"{init} b{budget} {CONDITION_NAMES[treatment]} - {CONDITION_NAMES[control]}: wrote {name}_*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("superclasses", help="write the CIFAR-100 class-to-superclass map")
    write.add_argument("--data-root", type=Path, default=Path("data"))
    write.add_argument("--output", type=Path, default=Path("src/component/configs/cifar100_superclasses.csv"))
    run = commands.add_parser("analyze", help="write the error-analysis tables and figures")
    run.add_argument("--root", type=Path, required=True, help="runs of one or more initializations")
    run.add_argument("--superclasses", type=Path, default=Path("src/component/configs/cifar100_superclasses.csv"))
    run.add_argument("--tables", type=Path, default=Path("reports/Latex_report/tables"))
    run.add_argument("--figures", type=Path, default=Path("reports/Latex_report/fig"))
    run.add_argument("--budget", default="5", help="data budget of the per-class analysis")
    run.add_argument("--top", type=int, default=10, help="confusions listed per direction")
    args = parser.parse_args()
    if args.command == "superclasses":
        print(f"wrote {write_superclasses(args.data_root, args.output)}")
    else:
        analyze(args.root, args.superclasses, args.tables, args.figures, args.budget, args.top)


if __name__ == "__main__":
    main()
