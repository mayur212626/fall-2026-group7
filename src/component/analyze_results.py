"""Aggregate training runs into result tables and figures.

Reads every run below ``--root`` (directories with ``config.json`` and
``result.json``), then prints:

- the mean and sample standard deviation over seeds of each model, data
  budget and condition, and
- the paired change of RandAugment over real-only for each model and data
  budget: the per-seed differences, their mean and standard deviation, and
  a verdict. A change counts as a gain or a loss only when every seed agrees
  in sign; otherwise it is inconclusive.

It also saves an accuracy-versus-data-size figure as SVG and PDF, and with
``--table`` both tables as CSV plus a LaTeX table (one per initialization).

    python -m src.component.analyze_results --root runs/stage1-pretrained \
        --figure reports/Latex_report/fig/stage1_pretrained_accuracy \
        --table reports/Latex_report/tables/stage1_pretrained_accuracy

Scores are the best validation scores of each run, or with ``--split test``
the one-time test scores written by ``evaluate_test``.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

BUDGET_ORDER = ["5", "10", "20", "50", "full"]
FULL_IMAGES_PER_CLASS = 450
MODEL_NAMES = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16"}
MODEL_COLORS = {"resnet50": "#2a78d6", "vit_b_16": "#eb6834"}  # categorical slots 1-2, validated
CONDITION_NAMES = {"real_only": "real-only", "randaugment": "RandAugment"}
CONDITION_LINES = {"real_only": "-", "randaugment": "--"}
METRIC_NAMES = {"accuracy": "top-1 accuracy", "macro_f1": "macro-F1", "balanced_accuracy": "balanced accuracy"}


def images_per_class(budget: str) -> int:
    """Training images per class of a data budget; ``"full"`` is 450."""
    return FULL_IMAGES_PER_CLASS if budget == "full" else int(budget)


def load_runs(root: Path, split: str = "validation") -> list[dict]:
    """Return one row per completed run below ``root`` with its settings and scores.

    ``split="validation"`` uses the best validation scores from ``result.json``;
    ``split="test"`` uses ``test_result.json`` and leaves out runs without one.
    """
    rows = []
    for result_path in sorted(Path(root).rglob("result.json")):
        run_dir = result_path.parent
        settings = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["settings"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if split == "test":
            test_path = run_dir / "test_result.json"
            if not test_path.exists():
                continue
            best = json.loads(test_path.read_text(encoding="utf-8"))["test"]
        else:
            best = result["best_validation"]
        rows.append({
            "init": settings["init"],
            "model": settings["model"],
            "condition": settings["condition"],
            "budget": settings["budget"],
            "seed": settings["seed"],
            "accuracy": best["accuracy"],
            "macro_f1": best["macro_f1"],
            "balanced_accuracy": best["balanced_accuracy"],
            "best_step": result["best_step"],
            "train_minutes": result["cost"]["train_seconds"] / 60,
        })
    return rows


def _budget_key(budget: str) -> int:
    return BUDGET_ORDER.index(budget) if budget in BUDGET_ORDER else len(BUDGET_ORDER)


def aggregate(rows: list[dict], metric: str) -> list[dict]:
    """Mean and sample standard deviation of ``metric`` over seeds, per group.

    Groups are (init, model, budget, condition). ``sd`` is None for one seed.
    """
    groups: dict[tuple, dict[int, float]] = {}
    for r in rows:
        key = (r["init"], r["model"], r["budget"], r["condition"])
        groups.setdefault(key, {})[r["seed"]] = r[metric]
    result = []
    for (init, model, budget, condition), by_seed in groups.items():
        values = [by_seed[seed] for seed in sorted(by_seed)]
        result.append({
            "init": init, "model": model, "budget": budget, "condition": condition,
            "n": len(values), "seeds": sorted(by_seed),
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
        })
    return sorted(result, key=lambda g: (g["init"], g["model"], _budget_key(g["budget"]), g["condition"]))


def paired_changes(rows: list[dict], treatment: str, control: str, metric: str) -> list[dict]:
    """Per-seed change of ``treatment`` over ``control`` for each (init, model, budget).

    Only seeds present in both conditions are paired.
    """
    values: dict[tuple, dict[str, dict[int, float]]] = {}
    for r in rows:
        key = (r["init"], r["model"], r["budget"])
        values.setdefault(key, {}).setdefault(r["condition"], {})[r["seed"]] = r[metric]
    result = []
    for (init, model, budget), by_condition in values.items():
        treated = by_condition.get(treatment, {})
        controls = by_condition.get(control, {})
        seeds = sorted(set(treated) & set(controls))
        if not seeds:
            continue
        changes = [treated[s] - controls[s] for s in seeds]
        if all(c > 0 for c in changes):
            verdict = "gain"
        elif all(c < 0 for c in changes):
            verdict = "loss"
        else:
            verdict = "inconclusive"
        result.append({
            "init": init, "model": model, "budget": budget, "seeds": seeds, "changes": changes,
            "mean": statistics.mean(changes),
            "sd": statistics.stdev(changes) if len(changes) > 1 else None,
            "verdict": verdict,
        })
    return sorted(result, key=lambda c: (c["init"], c["model"], _budget_key(c["budget"])))


def plot_accuracy(aggregates: list[dict], out_stem: Path, init: str, split: str = "validation") -> list[Path]:
    """Plot mean accuracy (±1 SD over seeds) against images per class.

    Color identifies the model and line style the condition. Saves
    ``out_stem.svg`` and ``out_stem.pdf`` and returns both paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    groups = [g for g in aggregates if g["init"] == init]
    for model in MODEL_NAMES:
        for condition in CONDITION_NAMES:
            series = sorted(
                (g for g in groups if g["model"] == model and g["condition"] == condition),
                key=lambda g: _budget_key(g["budget"]),
            )
            if not series:
                continue
            x = [images_per_class(g["budget"]) for g in series]
            y = [g["mean"] for g in series]
            err = [g["sd"] or 0.0 for g in series]
            label = f"{MODEL_NAMES[model]}, {CONDITION_NAMES[condition]}"
            ax.errorbar(
                x, y, yerr=err, label=label, color=MODEL_COLORS[model],
                linestyle=CONDITION_LINES[condition], linewidth=1.5, marker="o", markersize=5,
                capsize=3,
            )
    ticks = [images_per_class(b) for b in BUDGET_ORDER]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}" if t != FULL_IMAGES_PER_CLASS else f"full ({t})" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Training images per class (log scale)")
    ax.set_ylabel(f"{split.capitalize()} top-1 accuracy (%)")
    ax.grid(True, color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_stem.with_suffix(".svg"), out_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def write_csv(records: list[dict], path: Path) -> Path:
    """Write one CSV row per record; floats are rounded to 4 decimals and lists joined by spaces."""
    def cell(value: object) -> object:
        if isinstance(value, list):
            return " ".join(str(cell(v)) for v in value)
        return round(value, 4) if isinstance(value, float) else value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: cell(v) for k, v in record.items()})
    return path


def _tex(mean: float, sd: float | None, sign: str = "") -> str:
    return f"{mean:{sign}.2f}" if sd is None else f"{mean:{sign}.2f} $\\pm$ {sd:.2f}"


def latex_table(aggregates: list[dict], changes: list[dict], init: str, metric: str, split: str) -> str:
    """Booktabs table of real-only, RandAugment and their paired change for one initialization."""
    means = {(g["model"], g["budget"], g["condition"]): g for g in aggregates if g["init"] == init}
    paired = {(c["model"], c["budget"]): c for c in changes if c["init"] == init}
    seeds = sorted({g["n"] for g in means.values()})
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Stage 1 baselines, {init} initialization: {split} {METRIC_NAMES[metric]} (\%), mean $\pm$ "
        rf"sample SD over {' or '.join(map(str, seeds))} seeds. The change is RandAugment minus real-only per seed "
        r"(percentage points); it is a gain or a loss only when every seed agrees in sign.}",
        rf"\label{{tab:stage1-{init}-{metric}}}",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Model & Images/class & Real-only & RandAugment & Change & Verdict \\",
    ]
    for model in MODEL_NAMES:
        budgets = sorted({b for m, b, _ in means if m == model}, key=_budget_key)
        if not budgets:
            continue
        lines.append(r"\midrule")
        for budget in budgets:
            cells = [MODEL_NAMES[model], budget]
            for condition in CONDITION_NAMES:
                g = means.get((model, budget, condition))
                cells.append(_tex(g["mean"], g["sd"]) if g else "--")
            c = paired.get((model, budget))
            cells += [_tex(c["mean"], c["sd"], "+"), c["verdict"]] if c else ["--", "--"]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _fmt(mean: float, sd: float | None) -> str:
    return f"{mean:.2f}" if sd is None else f"{mean:.2f} ± {sd:.2f}"


def main() -> None:
    """Print the aggregate and paired-change tables and save the figure."""
    parser = argparse.ArgumentParser(description="Aggregate training runs.")
    parser.add_argument("--root", type=Path, default=Path("runs/stage1-pretrained"))
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "macro_f1", "balanced_accuracy"])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--figure", type=Path, help="figure path without extension; saves .svg and .pdf")
    parser.add_argument("--table", type=Path,
                        help="table path without extension; saves .csv, _paired.csv and LaTeX .tex")
    args = parser.parse_args()

    rows = load_runs(args.root, args.split)
    groups = aggregate(rows, args.metric)
    print(f"{len(rows)} runs, metric: {args.split} {args.metric}\n")
    if args.split == "test":
        missing = len(load_runs(args.root, "validation")) - len(rows)
        if missing:
            print(f"WARNING: {missing} completed runs have no test scores yet; the tables below leave them out.\n")
    print("| init | model | images/class | condition | seeds | mean ± SD |")
    print("|---|---|---|---|---|---|")
    for g in groups:
        print(f"| {g['init']} | {MODEL_NAMES.get(g['model'], g['model'])} | {g['budget']} | "
              f"{CONDITION_NAMES.get(g['condition'], g['condition'])} | {g['n']} | {_fmt(g['mean'], g['sd'])} |")

    print("\nPaired change, RandAugment minus real-only (percentage points):\n")
    print("| init | model | images/class | per-seed changes | mean ± SD | verdict |")
    print("|---|---|---|---|---|---|")
    changes = paired_changes(rows, "randaugment", "real_only", args.metric)
    for c in changes:
        per_seed = ", ".join(f"{v:+.2f}" for v in c["changes"])
        print(f"| {c['init']} | {MODEL_NAMES.get(c['model'], c['model'])} | {c['budget']} | {per_seed} | "
              f"{_fmt(c['mean'], c['sd'])} | {c['verdict']} |")

    if args.figure:
        inits = sorted({g["init"] for g in groups})
        for init in inits:
            stem = args.figure if len(inits) == 1 else args.figure.with_name(f"{args.figure.name}_{init}")
            for path in plot_accuracy(groups, stem, init, args.split):
                print(f"saved {path}")

    if args.table:
        print(f"saved {write_csv(groups, args.table.with_suffix('.csv'))}")
        print(f"saved {write_csv(changes, args.table.with_name(args.table.name + '_paired.csv'))}")
        inits = sorted({g["init"] for g in groups})
        for init in inits:
            stem = args.table.name if len(inits) == 1 else f"{args.table.name}_{init}"
            path = args.table.with_name(f"{stem}.tex")
            path.write_text(latex_table(groups, changes, init, args.metric, args.split), encoding="utf-8")
            print(f"saved {path}")


if __name__ == "__main__":
    main()
