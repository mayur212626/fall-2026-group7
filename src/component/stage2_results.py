"""Stage 2 results: ConvNeXt-Tiny, synthetic:real ratios, and Mixup and CutMix, each paired with its controls.

Each study answers one question about the sign flip (SD class-prompt images help
ImageNet-pretrained classifiers but lose to RandAugment from random weights):

    convnext   Does the flip hold for a third architecture?
    ratio      Does a larger share of generated images (2:1, 5:1) help pretrained
               classifiers more and randomly initialized ones less?
    mixing     Do the generated images lose to conventional augmentation in general
               from random weights, or only to RandAugment?

The runs come from several folders: the Stage 1 runs give the real-only,
RandAugment and 1:1 SD controls of ResNet-50 and ViT-B/16, and the Stage 2
folders the new runs. ``analyze_results.load_runs`` names a run at a ratio other
than 1:1 ``<condition>_r<ratio>``. Every comparison is a paired change by
training seed over the seeds both conditions have, with the Stage 1 verdict rule:
a gain or a loss only when every seed agrees in sign.

    python -m src.component.stage2_results --tables reports/Latex_report/tables \
        --figures reports/Latex_report/fig
    python -m src.component.stage2_results --split test ...   # after the test evaluation
"""

import argparse
from pathlib import Path

from src.component.analyze_results import (
    MODEL_COLORS,
    MODEL_NAMES,
    VERDICT_MARKS,
    _budget_key,
    _tex,
    aggregate,
    load_runs,
    paired_changes,
    write_csv,
)

ROOTS = ["runs/stage1-{init}", "runs/stage2-convnext-{init}", "runs/stage2-ratio-{init}", "runs/stage2-mixing-{init}"]
# study -> models, and the (treatment, control, column label) comparisons it reports
STUDIES = {
    "convnext": (["convnext_tiny"], [
        ("sd_prompt", "real_only", "SD$-$real"),
        ("randaugment", "real_only", "RA$-$real"),
        ("sd_prompt", "randaugment", "SD$-$RA"),
        ("sd_prompt_randaugment", "randaugment", "SD+RA$-$RA"),
    ]),
    "ratio": (["resnet50", "vit_b_16"], [
        ("sd_prompt", "real_only", "1:1$-$real"),
        ("sd_prompt_r2", "real_only", "2:1$-$real"),
        ("sd_prompt_r5", "real_only", "5:1$-$real"),
        ("sd_prompt", "randaugment", "1:1$-$RA"),
        ("sd_prompt_r2", "randaugment", "2:1$-$RA"),
        ("sd_prompt_r5", "randaugment", "5:1$-$RA"),
    ]),
    "mixing": (["resnet50", "vit_b_16"], [
        ("randaugment", "real_only", "RA$-$real"),
        ("mixup", "real_only", "Mixup$-$real"),
        ("cutmix", "real_only", "CutMix$-$real"),
        ("sd_prompt", "randaugment", "SD$-$RA"),
        ("sd_prompt", "mixup", "SD$-$Mixup"),
        ("sd_prompt", "cutmix", "SD$-$CutMix"),
    ]),
}
CAPTIONS = {
    "convnext": "ConvNeXt-T, the third architecture",
    "ratio": "SD class prompts at synthetic:real ratios 1:1, 2:1 and 5:1",
    "mixing": "SD class prompts against Mixup and CutMix",
}
# the treatment that only Stage 2 trained: a study lists only the cells where it has runs
STUDY_KEYS = {"convnext": "sd_prompt", "ratio": "sd_prompt_r2", "mixing": "mixup"}
RATIOS = {"sd_prompt": 1, "sd_prompt_r2": 2, "sd_prompt_r5": 5}


def load(init: str, split: str, roots: list[str]) -> list[dict]:
    """Runs of one initialization from every folder that exists (Stage 1 controls and Stage 2 runs)."""
    rows = []
    for pattern in roots:
        root = Path(pattern.format(init=init))
        if root.exists():
            rows += load_runs(root, split)
    return rows


def study_changes(rows: list[dict], study: str, metric: str) -> list[dict]:
    """Paired changes of a study's comparisons, only where the study's own Stage 2 condition has runs."""
    models, comparisons = STUDIES[study]
    rows = [r for r in rows if r["model"] in models]
    key = STUDY_KEYS[study]
    cells = {(r["model"], r["budget"]) for r in rows if r["condition"] == key}
    out = []
    for treatment, control, label in comparisons:
        for c in paired_changes(rows, treatment, control, metric):
            if (c["model"], c["budget"]) in cells:
                out.append({"study": study, "treatment": treatment, "control": control, "label": label, **c})
    return out


def study_table(changes: list[dict], study: str, init: str, metric: str, split: str) -> str:
    """Booktabs table of one study for one initialization: one column per comparison."""
    _, comparisons = STUDIES[study]
    labels = [label for _, _, label in comparisons]
    cell = {(c["model"], c["budget"], c["label"]): c for c in changes if c["init"] == init}
    seeds = sorted({len(c["seeds"]) for c in cell.values()})
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Stage 2, {CAPTIONS[study]}, {init} initialization: change in {split} "
        rf"{'top-1 accuracy' if metric == 'accuracy' else metric.replace('_', ' ')} (percentage points), paired by "
        rf"training seed, mean $\pm$ sample SD over {' or '.join(map(str, seeds))} seeds. SD: Stable Diffusion with "
        r"class prompts (1:1 unless a ratio is given); RA: RandAugment. $\uparrow$ gain and $\downarrow$ loss when "
        r"every seed agrees in sign, $\sim$ inconclusive otherwise.}",
        rf"\label{{tab:stage2-{study}-{init}{'-test' if split == 'test' else ''}}}",
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
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def plot_ratios(changes: list[dict], out_stem: Path, split: str) -> list[Path]:
    """Paired gain over real-only against the synthetic:real ratio; one panel per initialization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = [c for c in changes if c["study"] == "ratio" and c["control"] == "real_only"]
    inits = sorted({c["init"] for c in gains})
    budgets = sorted({c["budget"] for c in gains}, key=_budget_key)
    markers = dict(zip(budgets, ["o", "s", "^", "D", "v"], strict=False))
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(1, len(inits), figsize=(3.6 * len(inits), 3.2), squeeze=False)
    for ax, init in zip(axes[0], inits, strict=True):
        for model in MODEL_NAMES:
            for budget in budgets:
                pts = sorted((RATIOS[c["treatment"]], c["mean"], c["sd"] or 0.0) for c in gains
                             if c["init"] == init and c["model"] == model and c["budget"] == budget)
                if len(pts) < 2:
                    continue
                x, y, err = zip(*pts, strict=True)
                ax.errorbar(x, y, yerr=err, color=MODEL_COLORS[model], marker=markers[budget], markersize=5,
                            linewidth=1.2, capsize=2, label=f"{MODEL_NAMES[model]}, {budget}/class")
        ax.axhline(0, color="#8a8a86", linewidth=0.8)
        ax.set_xticks([1, 2, 5])
        ax.set_xticklabels(["1:1", "2:1", "5:1"])
        ax.set_xlabel("generated : real images")
        ax.set_title("ImageNet-pretrained" if init == "pretrained" else "random initialization", fontsize=9)
        ax.grid(True, color="#e0e0e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel(f"gain over real-only, {split} (pp)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(h)
                labels.append(label)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_stem.with_suffix(".svg"), out_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "macro_f1", "balanced_accuracy"])
    parser.add_argument("--roots", nargs="+", default=ROOTS, help="run folders, with {init} for the initialization")
    parser.add_argument("--tables", type=Path, default=Path("reports/Latex_report/tables"))
    parser.add_argument("--figures", type=Path, default=Path("reports/Latex_report/fig"))
    args = parser.parse_args()

    suffix = "_test" if args.split == "test" else ""
    all_changes, all_means = [], []
    for init in ("pretrained", "scratch"):
        rows = load(init, args.split, args.roots)
        if not rows:
            continue
        all_means += [g for g in aggregate(rows, args.metric)
                      if g["model"] == "convnext_tiny" or g["condition"] in ("sd_prompt_r2", "sd_prompt_r5", "mixup",
                                                                              "cutmix")]
        for study in STUDIES:
            changes = study_changes(rows, study, args.metric)
            if not changes:
                continue
            all_changes += changes
            path = args.tables / f"stage2_{study}_{init}{suffix}.tex"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(study_table(changes, study, init, args.metric, args.split), encoding="utf-8")
            print(f"wrote {path}")
    if not all_changes:
        raise SystemExit("no Stage 2 comparisons: check the run folders")
    print(f"wrote {write_csv(all_means, args.tables / f'stage2_accuracy{suffix}.csv')}")
    print(f"wrote {write_csv(all_changes, args.tables / f'stage2_paired{suffix}.csv')}")
    for path in plot_ratios(all_changes, args.figures / f"stage2_ratio_gain{suffix}", args.split):
        print(f"wrote {path}")

    for study in STUDIES:
        print(f"\n{study}: paired change (pp), mean over seeds, verdict")
        for c in all_changes:
            if c["study"] == study:
                print(f"  {c['init']:<10} {MODEL_NAMES[c['model']]:<10} b{c['budget']:<4} {c['label']:<16} "
                      f"{c['mean']:+6.2f}  ({len(c['seeds'])} seeds, {c['verdict']})")


if __name__ == "__main__":
    main()
