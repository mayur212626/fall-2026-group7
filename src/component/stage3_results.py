"""Stage 3 results: Flowers-102 and HAM10000, each paired with its controls.

For every dataset and initialization three tables, with the verdict rule of
Stages 1 and 2 (a gain or a loss only when every seed agrees in sign):

    scores        real-only, the best conventional augmentation, SD 1:1 and
                  SD 5:1 + RandAugment ("ours"), in %, and the paired change of
                  ours over the best conventional one
    conventional  each conventional augmentation against real-only
    sd            SD-based conditions against their controls

The best conventional augmentation (RandAugment, TrivialAugment, AutoAugment,
AugMix, Mixup or CutMix) of each model and budget is chosen by mean validation
balanced accuracy, the Stage 3 checkpoint metric, so the test tables compare
the conditions the validation set chose. Scores are balanced accuracy unless
``--metric`` says otherwise.

    python -m src.component.stage3_results --dataset ham10000
    python -m src.component.stage3_results --dataset ham10000 --split test    # after the test evaluation
"""

import argparse
from pathlib import Path

from src.component.analyze_results import (
    MODEL_NAMES,
    VERDICT_MARKS,
    _tex,
    aggregate,
    load_runs,
    paired_changes,
    write_csv,
)
from src.component.stage3_data import DATASETS, SELECTION_METRIC

ROOT = "runs/{dataset}-{init}"
MODELS = ["resnet50", "vit_b_16"]
CONVENTIONAL = ["randaugment", "trivialaugment", "autoaugment", "augmix", "mixup", "cutmix"]
OURS = "sd_prompt_randaugment_r5"
BEST = "best_conventional"
SHORT = {"real_only": "real", "randaugment": "RA", "trivialaugment": "TA", "autoaugment": "AA", "augmix": "AugMix",
         "mixup": "Mixup", "cutmix": "CutMix", "sd_prompt": "SD", "sd_prompt_randaugment": "SD+RA",
         OURS: "SD 5:1+RA", BEST: "best conv."}
STUDIES = {
    "conventional": [(c, "real_only", f"{SHORT[c]}$-$real") for c in CONVENTIONAL],
    "sd": [
        ("sd_prompt", "real_only", "SD$-$real"),
        ("sd_prompt", "randaugment", "SD$-$RA"),
        ("sd_prompt_randaugment", "randaugment", "SD+RA$-$RA"),
        (OURS, "randaugment", "SD 5:1+RA$-$RA"),
        (OURS, BEST, "SD 5:1+RA$-$best conv."),
    ],
}
CAPTIONS = {
    "conventional": "conventional augmentation against real-only",
    "sd": "Stable Diffusion class-prompt images against their controls",
}
METRIC_WORDS = {"balanced_accuracy": "balanced accuracy", "accuracy": "top-1 accuracy", "macro_f1": "macro-F1"}
MIN_SEEDS = 3


def _key(budget: str) -> int:
    return int(budget)


def choose_conventional(validation_rows: list[dict]) -> dict[tuple[str, str, str], str]:
    """Best conventional condition of every (init, model, budget) by mean validation balanced accuracy.

    A condition needs at least ``MIN_SEEDS`` seeds; a tie keeps the earlier condition in ``CONVENTIONAL``.
    """
    chosen = {}
    groups = {(g["init"], g["model"], g["budget"], g["condition"]): g
              for g in aggregate(validation_rows, SELECTION_METRIC)}
    for cell in sorted({k[:3] for k in groups}):
        candidates = [(groups[(*cell, c)]["mean"], -i, c) for i, c in enumerate(CONVENTIONAL)
                      if (*cell, c) in groups and groups[(*cell, c)]["n"] >= MIN_SEEDS]
        if candidates:
            chosen[cell] = max(candidates)[2]
    return chosen


def study_changes(rows: list[dict], study: str, metric: str, best: dict) -> list[dict]:
    """Paired changes of one study; ``best_conventional`` stands for the condition chosen in that cell."""
    out = []
    for treatment, control, label in STUDIES[study]:
        if control != BEST:
            found = paired_changes(rows, treatment, control, metric)
        else:
            found = []
            for (init, model, budget), condition in best.items():
                cell = [r for r in rows if (r["init"], r["model"], r["budget"]) == (init, model, budget)]
                found += [{**c, "best": condition} for c in paired_changes(cell, treatment, condition, metric)]
        out += [{"study": study, "treatment": treatment, "control": control, "label": label, "best": "", **c}
                for c in found]
    return sorted(out, key=lambda c: (c["init"], MODELS.index(c["model"]), _key(c["budget"])))


def _caption_tail(cells: list[dict], metric: str, split: str) -> str:
    seeds = sorted({len(c["seeds"]) for c in cells})
    return (rf"change in {split} {METRIC_WORDS[metric]} (percentage points), paired by training seed, mean $\pm$ "
            rf"sample SD over {' or '.join(map(str, seeds))} seeds. $\uparrow$ gain and $\downarrow$ loss when every "
            r"seed agrees in sign, $\sim$ inconclusive otherwise.")


def _label(dataset: str, table: str, init: str, split: str) -> str:
    return f"tab:stage3-{dataset}-{table}-{init}{'-test' if split == 'test' else ''}"


def study_table(changes: list[dict], dataset: str, study: str, init: str, metric: str, split: str) -> str:
    """Booktabs table of one study: one row per model and budget, one column per comparison."""
    labels = [label for _, _, label in STUDIES[study]]
    cell = {(c["model"], c["budget"], c["label"]): c for c in changes if c["init"] == init and c["study"] == study}
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{Stage 3, {DATASETS[dataset]['display']}, {CAPTIONS[study]}, {init} initialization: "
        + _caption_tail(list(cell.values()), metric, split)
        + r" RA: RandAugment, TA: TrivialAugment, AA: AutoAugment (ImageNet policy); SD: Stable Diffusion class "
          r"prompts at 1:1 unless a ratio is given; best conv.: the conventional augmentation with the highest "
          r"mean validation balanced accuracy in that row (named in the scores table).}",
        rf"\label{{{_label(dataset, study, init, split)}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(labels) + "}",
        r"\toprule",
        "Model & Images/class & " + " & ".join(labels) + r" \\",
    ]
    for model in MODELS:
        budgets = sorted({b for m, b, _ in cell if m == model}, key=_key)
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


def scores_table(means: list[dict], changes: list[dict], best: dict, dataset: str, init: str, metric: str,
                 split: str) -> str:
    """Real-only, best conventional, SD 1:1 and ours in %, and the paired change of ours over best conventional."""
    score = {(g["model"], g["budget"], g["condition"]): g for g in means if g["init"] == init}
    gain = {(c["model"], c["budget"]): c for c in changes
            if c["init"] == init and c["treatment"] == OURS and c["control"] == BEST}

    def value(model: str, budget: str, condition: str | None) -> str:
        g = score.get((model, budget, condition))
        return _tex(g["mean"], g["sd"]) if g else "--"

    cells = sorted({(m, b) for (i, m, b) in best if i == init}, key=lambda mb: (MODELS.index(mb[0]), _key(mb[1])))
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Stage 3, {DATASETS[dataset]['display']}, {init} initialization: {split} "
        rf"{METRIC_WORDS[metric]} (\%, mean $\pm$ SD over seeds) of real-only, the best conventional augmentation "
        r"(chosen by mean validation balanced accuracy), SD class prompts at 1:1, and SD 5:1 + RandAugment, and the "
        r"paired change of SD 5:1 + RandAugment over the best conventional augmentation (percentage points; "
        r"$\uparrow$/$\downarrow$ when every seed agrees, $\sim$ otherwise).}",
        rf"\label{{{_label(dataset, 'scores', init, split)}}}",
        r"\begin{tabular}{llrlrrrr}",
        r"\toprule",
        r"Model & Images/class & Real-only & \multicolumn{2}{l}{Best conventional} & SD 1:1 & SD 5:1+RA & Change \\",
        r"\midrule",
    ]
    for model, budget in cells:
        conventional = best[(init, model, budget)]
        c = gain.get((model, budget))
        lines.append(" & ".join([
            MODEL_NAMES[model], budget, value(model, budget, "real_only"), SHORT[conventional],
            value(model, budget, conventional), value(model, budget, "sd_prompt"), value(model, budget, OURS),
            f"{_tex(c['mean'], c['sd'], '+')} {VERDICT_MARKS[c['verdict']]}" if c else "--",
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--metric", default=SELECTION_METRIC, choices=list(METRIC_WORDS))
    parser.add_argument("--root", default=ROOT, help="run folder, with {dataset} and {init}")
    parser.add_argument("--tables", type=Path, default=Path("reports/Latex_report/tables"))
    args = parser.parse_args()

    suffix = "_test" if args.split == "test" else ""
    all_changes, all_means = [], []
    for init in ("pretrained", "scratch"):
        root = Path(args.root.format(dataset=args.dataset, init=init))
        if not root.exists():
            continue
        rows = [r for r in load_runs(root, args.split) if r["model"] in MODELS]
        if not rows:
            continue
        validation = rows if args.split == "validation" else load_runs(root, "validation")
        best = choose_conventional(validation)
        means = aggregate(rows, args.metric)
        all_means += means
        changes = []
        for study in STUDIES:
            found = study_changes(rows, study, args.metric, best)
            if not found:
                continue
            changes += found
            path = args.tables / f"stage3_{args.dataset}_{study}_{init}{suffix}.tex"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(study_table(found, args.dataset, study, init, args.metric, args.split), encoding="utf-8")
            print(f"wrote {path}")
        if best:
            path = args.tables / f"stage3_{args.dataset}_scores_{init}{suffix}.tex"
            path.write_text(scores_table(means, changes, best, args.dataset, init, args.metric, args.split),
                            encoding="utf-8")
            print(f"wrote {path}")
        all_changes += changes
    if not all_changes:
        raise SystemExit(f"no Stage 3 comparisons for {args.dataset}: check {args.root}")
    print(f"wrote {write_csv(all_changes, args.tables / f'stage3_{args.dataset}_paired{suffix}.csv')}")
    print(f"wrote {write_csv(all_means, args.tables / f'stage3_{args.dataset}_scores{suffix}.csv')}")

    print(f"\n{DATASETS[args.dataset]['display']}, {args.split} {METRIC_WORDS[args.metric]}: paired change (pp)")
    for c in all_changes:
        if c["study"] == "sd":
            vs = f" (vs {SHORT[c['best']]})" if c["best"] else ""
            print(f"  {c['init']:<10} {MODEL_NAMES[c['model']]:<10} b{c['budget']:<3} {c['label']:<26} "
                  f"{c['mean']:+6.2f}  ({len(c['seeds'])} seeds, {c['verdict']}){vs}")


if __name__ == "__main__":
    main()
