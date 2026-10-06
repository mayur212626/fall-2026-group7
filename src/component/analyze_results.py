"""Aggregate training runs into result tables and figures.

Reads every run below ``--root`` (directories with ``config.json`` and
``result.json``), then prints:

- the mean and sample standard deviation over seeds of each model, data
  budget and condition, and
- the paired change of every other condition (RandAugment and the synthetic
  conditions) over real-only for each model and data budget: the per-seed
  differences, their mean and standard deviation, and a verdict. A change counts as a gain or a loss only when every seed agrees
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
CONDITION_NAMES = {
    "real_only": "real-only",
    "randaugment": "RandAugment",
    "sd_prompt": "SD class prompts",
    "sd_lora": "SD + LoRA",
    "sd_prompt_randaugment": "SD class prompts + RandAugment",
}
CONDITION_LINES = {"real_only": "-", "randaugment": "--", "sd_prompt": ":", "sd_lora": "-.",
                   "sd_prompt_randaugment": (0, (3, 1, 1, 1, 1, 1))}
CONDITION_MARKERS = {"real_only": "o", "randaugment": "o", "sd_prompt": "^", "sd_lora": "v",
                     "sd_prompt_randaugment": "s"}
# Synthetic conditions compared with RandAugment, the strongest real-data control.
RANDAUGMENT_COMPARISONS = {"sd_prompt": "SD$-$RA", "sd_lora": "LoRA$-$RA", "sd_prompt_randaugment": "SD+RA$-$RA"}
VERDICT_MARKS = {"gain": r"$\uparrow$", "loss": r"$\downarrow$", "inconclusive": r"$\sim$"}
CONTROL = "real_only"
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


def series_points(
    groups: list[dict], model: str, condition: str, budgets: list[str]
) -> tuple[list[int], list[float], list[float]]:
    """Images per class, means and SDs of one series at ``budgets``; NaN where it has no runs.

    The NaN gaps break the plotted line, so it never bridges budgets without data.
    """
    by_budget = {g["budget"]: g for g in groups if g["model"] == model and g["condition"] == condition}
    nan = float("nan")
    return (
        [images_per_class(b) for b in budgets],
        [by_budget[b]["mean"] if b in by_budget else nan for b in budgets],
        [(by_budget[b]["sd"] or 0.0) if b in by_budget else nan for b in budgets],
    )


def plot_accuracy(aggregates: list[dict], out_stem: Path, init: str, split: str = "validation") -> list[Path]:
    """Plot mean accuracy (±1 SD over seeds) against images per class.

    Color identifies the model; line style and marker identify the condition.
    With more than four series the legend moves to the right of the plot so it
    covers no data. Saves ``out_stem.svg`` and ``out_stem.pdf`` and returns both paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    groups = [g for g in aggregates if g["init"] == init]
    budgets = sorted({g["budget"] for g in groups}, key=_budget_key)
    series = [(m, c) for m in MODEL_NAMES for c in CONDITION_NAMES
              if any(g["model"] == m and g["condition"] == c for g in groups)]
    legend_outside = len(series) > 4
    fig, ax = plt.subplots(figsize=(8.4 if legend_outside else 6.0, 3.8))
    for model, condition in series:
        x, y, err = series_points(groups, model, condition, budgets)
        label = f"{MODEL_NAMES[model]}, {CONDITION_NAMES[condition]}"
        ax.errorbar(
            x, y, yerr=err, label=label, color=MODEL_COLORS[model],
            linestyle=CONDITION_LINES[condition], linewidth=1.5, marker=CONDITION_MARKERS[condition], markersize=5,
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
    if legend_outside:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
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


def latex_table(
    aggregates: list[dict], changes: list[dict], init: str, metric: str, split: str,
    treatment: str = "randaugment",
) -> str:
    """Booktabs table of real-only, ``treatment`` and their paired change for one initialization.

    ``changes`` must be the paired changes of ``treatment`` over real-only.
    """
    means = {(g["model"], g["budget"], g["condition"]): g for g in aggregates if g["init"] == init}
    paired = {(c["model"], c["budget"]): c for c in changes if c["init"] == init}
    seeds = sorted({g["n"] for g in means.values()})
    title = " baselines" if treatment == "randaugment" else f", {CONDITION_NAMES[treatment]} against real-only"
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Stage 1{title}, {init} initialization: {split} {METRIC_NAMES[metric]} (\%), mean $\pm$ "
        rf"sample SD over {' or '.join(map(str, seeds))} seeds. The change is {CONDITION_NAMES[treatment]} minus "
        r"real-only per seed (percentage points); it is a gain or a loss only when every seed agrees in sign.}",
        rf"\label{{tab:stage1-{init}-{metric}{'' if treatment == 'randaugment' else '-' + treatment}}}",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        rf"Model & Images/class & Real-only & {CONDITION_NAMES[treatment]} & Change & Verdict \\",
    ]
    for model in MODEL_NAMES:
        budgets = sorted({b for m, b, _ in means if m == model}, key=_budget_key)
        if not budgets:
            continue
        lines.append(r"\midrule")
        for budget in budgets:
            cells = [MODEL_NAMES[model], budget]
            for condition in (CONTROL, treatment):
                g = means.get((model, budget, condition))
                cells.append(_tex(g["mean"], g["sd"]) if g else "--")
            c = paired.get((model, budget))
            cells += [_tex(c["mean"], c["sd"], "+"), c["verdict"]] if c else ["--", "--"]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def randaugment_comparison_table(rows: list[dict], init: str, metric: str, split: str) -> str:
    """Booktabs table of each synthetic condition minus RandAugment, paired by seed, for one initialization.

    Lists every model and data budget with synthetic runs; "--" marks a
    condition without runs there.
    """
    rows = [r for r in rows if r["init"] == init]
    pairs = {t: {(c["model"], c["budget"]): c for c in paired_changes(rows, t, "randaugment", metric)}
             for t in RANDAUGMENT_COMPARISONS}
    seeds = sorted({len(c["seeds"]) for pair in pairs.values() for c in pair.values()})
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Stage 1 synthetic conditions against RandAugment, {init} initialization: change in {split} "
        rf"{METRIC_NAMES[metric]} (percentage points), paired by training seed, mean $\pm$ sample SD over "
        rf"{' or '.join(map(str, seeds))} seeds. SD: pretrained Stable Diffusion with class prompts; LoRA: "
        r"LoRA-adapted Stable Diffusion; RA: RandAugment. $\uparrow$ gain and $\downarrow$ loss when every seed "
        r"agrees in sign, $\sim$ inconclusive otherwise.}",
        rf"\label{{tab:stage1-{init}-{metric}-vs-randaugment}}",
        r"\begin{tabular}{ll" + "r" * len(RANDAUGMENT_COMPARISONS) + "}",
        r"\toprule",
        "Model & Images/class & " + " & ".join(RANDAUGMENT_COMPARISONS.values()) + r" \\",
    ]
    for model in MODEL_NAMES:
        budgets = sorted({b for pair in pairs.values() for m, b in pair if m == model}, key=_budget_key)
        if not budgets:
            continue
        lines.append(r"\midrule")
        for budget in budgets:
            cells = [MODEL_NAMES[model], budget]
            for pair in pairs.values():
                c = pair.get((model, budget))
                cells.append(f"{_tex(c['mean'], c['sd'], '+')} {VERDICT_MARKS[c['verdict']]}" if c else "--")
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

    present = {r["condition"] for r in rows}
    treatments = [c for c in CONDITION_NAMES if c != CONTROL and c in present]
    changes_by_treatment = {}
    for treatment in treatments:
        print(f"\nPaired change, {CONDITION_NAMES[treatment]} minus real-only (percentage points):\n")
        print("| init | model | images/class | per-seed changes | mean ± SD | verdict |")
        print("|---|---|---|---|---|---|")
        changes = paired_changes(rows, treatment, CONTROL, args.metric)
        changes_by_treatment[treatment] = changes
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
        # RandAugment keeps the original file names; other treatments get a suffix.
        for treatment, changes in changes_by_treatment.items():
            if not changes:
                continue
            suffix = "" if treatment == "randaugment" else f"_{treatment}"
            paired = changes if treatment == "randaugment" else [
                {"treatment": treatment, "control": CONTROL, **c} for c in changes
            ]
            print(f"saved {write_csv(paired, args.table.with_name(args.table.name + suffix + '_paired.csv'))}")
            inits = sorted({g["init"] for g in groups})
            for init in inits:
                stem = args.table.name if len(inits) == 1 else f"{args.table.name}_{init}"
                path = args.table.with_name(f"{stem}{suffix}.tex")
                path.write_text(latex_table(groups, changes, init, args.metric, args.split, treatment), encoding="utf-8")
                print(f"saved {path}")

    comparisons = [{"treatment": t, "control": "randaugment", **c}
                   for t in RANDAUGMENT_COMPARISONS for c in paired_changes(rows, t, "randaugment", args.metric)]
    if comparisons:
        print("\nPaired change against RandAugment (percentage points):\n")
        print("| init | model | images/class | condition | per-seed changes | mean ± SD | verdict |")
        print("|---|---|---|---|---|---|---|")
        for c in comparisons:
            per_seed = ", ".join(f"{v:+.2f}" for v in c["changes"])
            print(f"| {c['init']} | {MODEL_NAMES.get(c['model'], c['model'])} | {c['budget']} | "
                  f"{CONDITION_NAMES[c['treatment']]} | {per_seed} | {_fmt(c['mean'], c['sd'])} | {c['verdict']} |")
    if comparisons and args.table:
        print(f"saved {write_csv(comparisons, args.table.with_name(args.table.name + '_vs_randaugment_paired.csv'))}")
        inits = sorted({c["init"] for c in comparisons})
        for init in inits:
            stem = args.table.name if len(inits) == 1 else f"{args.table.name}_{init}"
            path = args.table.with_name(f"{stem}_vs_randaugment.tex")
            path.write_text(randaugment_comparison_table(rows, init, args.metric, args.split), encoding="utf-8")
            print(f"saved {path}")


if __name__ == "__main__":
    main()
