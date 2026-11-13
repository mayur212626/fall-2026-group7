"""Stacked-augmentation results: conventional policies, SD images on top of them, and the best of each family.

Three tables per initialization, every comparison a paired change by training seed
with the verdict rule of Stages 1 and 2 (a gain or a loss only when every seed agrees):

    policies   Which conventional augmentation is strongest on its own?
               RandAugment, TrivialAugment, AutoAugment and AugMix against real-only.
    stacked    Do SD class-prompt images still add accuracy on top of each one?
               Each policy with SD images against the same policy without them,
               plus SD + RandAugment + CutMix and SD 5:1 + RandAugment against RandAugment.
    best       Best SD-based condition against the best conventional one, for every
               model and budget. Both are chosen by mean validation accuracy only, so
               the test table compares the same two conditions the validation table chose.

The runs come from the Stage 1 and Stage 2 folders (real-only, RandAugment, SD, SD +
RandAugment, Mixup, CutMix, ratios) and ``runs/augment-<init>`` (the new conditions).

    python -m src.component.augment_results
    python -m src.component.augment_results --split test     # after the test evaluation
"""

import argparse
from pathlib import Path

from src.component.analyze_results import (
    MODEL_NAMES,
    VERDICT_MARKS,
    _budget_key,
    _tex,
    aggregate,
    paired_changes,
    write_csv,
)
from src.component.stage2_results import load

ROOTS = ["runs/stage1-{init}", "runs/stage2-ratio-{init}", "runs/stage2-mixing-{init}", "runs/augment-{init}"]
MODELS = ["resnet50", "vit_b_16"]
STUDIES = {
    "policies": [
        ("randaugment", "real_only", "RA$-$real"),
        ("trivialaugment", "real_only", "TA$-$real"),
        ("autoaugment", "real_only", "AA$-$real"),
        ("augmix", "real_only", "AugMix$-$real"),
    ],
    "stacked": [
        ("sd_prompt_randaugment", "randaugment", "SD+RA$-$RA"),
        ("sd_prompt_trivialaugment", "trivialaugment", "SD+TA$-$TA"),
        ("sd_prompt_autoaugment", "autoaugment", "SD+AA$-$AA"),
        ("sd_prompt_augmix", "augmix", "SD+AugMix$-$AugMix"),
        ("sd_prompt_randaugment_cutmix", "randaugment", "SD+RA+CutMix$-$RA"),
        ("sd_prompt_randaugment_r5", "randaugment", "SD 5:1+RA$-$RA"),
    ],
}
CAPTIONS = {
    "policies": "conventional augmentation policies against real-only",
    "stacked": "SD class-prompt images added on top of each conventional augmentation",
}
# the condition only this study trained: a table lists only the cells where it has runs
STUDY_KEYS = {"policies": "trivialaugment", "stacked": "sd_prompt_trivialaugment"}
CONVENTIONAL = ["randaugment", "trivialaugment", "autoaugment", "augmix", "mixup", "cutmix"]
SD_BASED = ["sd_prompt", "sd_prompt_r2", "sd_prompt_r5", "sd_prompt_randaugment", "sd_prompt_randaugment_r5",
            "sd_prompt_trivialaugment", "sd_prompt_autoaugment", "sd_prompt_augmix", "sd_prompt_cutmix",
            "sd_prompt_randaugment_cutmix"]
SHORT = {"real_only": "real", "randaugment": "RA", "trivialaugment": "TA", "autoaugment": "AA", "augmix": "AugMix",
         "mixup": "Mixup", "cutmix": "CutMix", "sd_prompt": "SD", "sd_prompt_r2": "SD 2:1", "sd_prompt_r5": "SD 5:1",
         "sd_prompt_randaugment": "SD+RA", "sd_prompt_randaugment_r5": "SD 5:1+RA",
         "sd_prompt_trivialaugment": "SD+TA", "sd_prompt_autoaugment": "SD+AA", "sd_prompt_augmix": "SD+AugMix",
         "sd_prompt_cutmix": "SD+CutMix", "sd_prompt_randaugment_cutmix": "SD+RA+CutMix"}
MIN_SEEDS = 3


def study_changes(rows: list[dict], study: str, metric: str) -> list[dict]:
    """Paired changes of a study's comparisons, only where the study's own new condition has runs."""
    rows = [r for r in rows if r["model"] in MODELS]
    cells = {(r["model"], r["budget"]) for r in rows if r["condition"] == STUDY_KEYS[study]}
    out = []
    for treatment, control, label in STUDIES[study]:
        for c in paired_changes(rows, treatment, control, metric):
            if (c["model"], c["budget"]) in cells:
                out.append({"study": study, "treatment": treatment, "control": control, "label": label, **c})
    return out


def choose_best(validation_rows: list[dict], metric: str) -> dict[tuple[str, str, str], dict[str, str]]:
    """Best conventional and best SD-based condition of every (init, model, budget), by mean validation score.

    A condition is a candidate only with at least ``MIN_SEEDS`` seeds; ties keep the earlier condition
    in ``CONVENTIONAL`` or ``SD_BASED``.
    """
    means = {(g["init"], g["model"], g["budget"], g["condition"]): g for g in aggregate(validation_rows, metric)}
    chosen: dict[tuple[str, str, str], dict[str, str]] = {}
    for init, model, budget in sorted({k[:3] for k in means}):
        pick = {}
        for family, names in (("conventional", CONVENTIONAL), ("sd", SD_BASED)):
            candidates = [(means[(init, model, budget, c)]["mean"], -i, c) for i, c in enumerate(names)
                          if (init, model, budget, c) in means and means[(init, model, budget, c)]["n"] >= MIN_SEEDS]
            if candidates:
                pick[family] = max(candidates)[2]
        if len(pick) == 2:
            chosen[(init, model, budget)] = pick
    return chosen


def best_changes(rows: list[dict], chosen: dict, metric: str) -> list[dict]:
    """Paired change of the chosen best SD-based condition over the chosen best conventional one."""
    out = []
    for (init, model, budget), pick in chosen.items():
        cell = [r for r in rows if (r["init"], r["model"], r["budget"]) == (init, model, budget)]
        for c in paired_changes(cell, pick["sd"], pick["conventional"], metric):
            out.append({"study": "best", "treatment": pick["sd"], "control": pick["conventional"],
                        "label": f"{SHORT[pick['sd']]}$-${SHORT[pick['conventional']]}", **c})
    return out


def _caption_tail(cells: list[dict], metric: str, split: str) -> str:
    seeds = sorted({len(c["seeds"]) for c in cells})
    what = "top-1 accuracy" if metric == "accuracy" else metric.replace("_", " ")
    return (rf"change in {split} {what} (percentage points), paired by training seed, mean $\pm$ sample SD over "
            rf"{' or '.join(map(str, seeds))} seeds. $\uparrow$ gain and $\downarrow$ loss when every seed agrees "
            r"in sign, $\sim$ inconclusive otherwise.")


def _join(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def study_table(changes: list[dict], study: str, init: str, metric: str, split: str) -> str:
    """Booktabs table of one study for one initialization: one column per comparison."""
    labels = [label for _, _, label in STUDIES[study]]
    cell = {(c["model"], c["budget"], c["label"]): c for c in changes if c["init"] == init}
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{Augmentation, {CAPTIONS[study]}, {init} initialization: "
        + _caption_tail(list(cell.values()), metric, split)
        + r" RA: RandAugment, TA: TrivialAugment, AA: AutoAugment (CIFAR-10 policy), SD: Stable Diffusion class "
          r"prompts at 1:1 unless a ratio is given.}",
        rf"\label{{tab:augment-{study}-{init}{'-test' if split == 'test' else ''}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(labels) + "}",
        r"\toprule",
        "Model & Images/class & " + " & ".join(labels) + r" \\",
    ]
    for model in MODELS:
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


def best_table(changes: list[dict], means: list[dict], init: str, metric: str, split: str) -> str:
    """Best conventional against best SD-based condition, chosen on validation, for one initialization."""
    score = {(g["model"], g["budget"], g["condition"]): g for g in means if g["init"] == init}
    rows = sorted((c for c in changes if c["init"] == init),
                  key=lambda c: (list(MODEL_NAMES).index(c["model"]), _budget_key(c["budget"])))
    # budgets where this study's new conditions were not run: the choice there has fewer candidates
    fewer = sorted({c["budget"] for c in rows
                    if (c["model"], c["budget"], STUDY_KEYS["policies"]) not in score}, key=_budget_key)
    note = (rf" At {_join(fewer)} images per class TrivialAugment, AutoAugment, AugMix and the stacked conditions "
            r"were not run, so the best conditions there are chosen from fewer candidates." if fewer else "")
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Best SD-based against best conventional augmentation, {init} initialization, both chosen by "
        rf"mean validation accuracy: {split} accuracy (\%, mean $\pm$ SD) of real-only and of the two chosen "
        r"conditions, and the " + _caption_tail(rows, metric, split)
        + r" The accuracies average every seed of each condition and the change only the seeds both conditions "
          r"share, so where their seeds differ the two need not subtract exactly." + note + "}",
        rf"\label{{tab:augment-best-{init}{'-test' if split == 'test' else ''}}}",
        r"\begin{tabular}{llrlrlrr}",
        r"\toprule",
        r"Model & Images/class & Real-only & Best conventional & & Best SD-based & & Change \\",
        r"\midrule",
    ]
    for c in rows:
        real = score.get((c["model"], c["budget"], "real_only"))
        conv, sd = score[(c["model"], c["budget"], c["control"])], score[(c["model"], c["budget"], c["treatment"])]
        lines.append(" & ".join([
            MODEL_NAMES[c["model"]], c["budget"], _tex(real["mean"], real["sd"]) if real else "--",
            SHORT[c["control"]], _tex(conv["mean"], conv["sd"]), SHORT[c["treatment"]], _tex(sd["mean"], sd["sd"]),
            f"{_tex(c['mean'], c['sd'], '+')} {VERDICT_MARKS[c['verdict']]}",
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "macro_f1", "balanced_accuracy"])
    parser.add_argument("--roots", nargs="+", default=ROOTS, help="run folders, with {init} for the initialization")
    parser.add_argument("--tables", type=Path, default=Path("reports/Latex_report/tables"))
    args = parser.parse_args()

    suffix = "_test" if args.split == "test" else ""
    all_changes, all_means = [], []
    for init in ("pretrained", "scratch"):
        rows = [r for r in load(init, args.split, args.roots) if r["model"] in MODELS]
        if not rows:
            continue
        validation = rows if args.split == "validation" else [
            r for r in load(init, "validation", args.roots) if r["model"] in MODELS]
        means = aggregate(rows, args.metric)
        all_means += means
        for study in STUDIES:
            changes = study_changes(rows, study, args.metric)
            if changes:
                all_changes += changes
                path = args.tables / f"augment_{study}_{init}{suffix}.tex"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(study_table(changes, study, init, args.metric, args.split), encoding="utf-8")
                print(f"wrote {path}")
        best = best_changes(rows, choose_best(validation, args.metric), args.metric)
        if best:
            all_changes += best
            path = args.tables / f"augment_best_{init}{suffix}.tex"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(best_table(best, means, init, args.metric, args.split), encoding="utf-8")
            print(f"wrote {path}")
    if not all_changes:
        raise SystemExit("no augmentation comparisons: check the run folders")
    print(f"wrote {write_csv(all_changes, args.tables / f'augment_paired{suffix}.csv')}")
    print(f"wrote {write_csv(all_means, args.tables / f'augment_accuracy{suffix}.csv')}")

    for study in (*STUDIES, "best"):
        print(f"\n{study}: paired change (pp), mean over seeds, verdict")
        for c in all_changes:
            if c["study"] == study:
                print(f"  {c['init']:<10} {MODEL_NAMES[c['model']]:<10} b{c['budget']:<4} {c['label']:<20} "
                      f"{c['mean']:+6.2f}  ({len(c['seeds'])} seeds, {c['verdict']})")


if __name__ == "__main__":
    main()
