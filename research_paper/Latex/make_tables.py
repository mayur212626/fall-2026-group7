"""Compact result tables for the paper, built from the test-set CSVs of the report.

The report's tables give every comparison of one initialization in its own wide table, with two
decimals and a full legend in each caption; at the paper's text width the widest ones shrink until
they cannot be read. This script writes one table per study with both initializations, one decimal
place and a short caption; the legend is given once in the Results section. Cells with three
training seeds are marked with a dagger (the others have five).

Run from the repository root after the test tables are on main:

    python research_paper/Latex/make_tables.py

The tables go to research_paper/Latex/tables/paper_<study>.tex.
"""

import csv
from pathlib import Path

SOURCE = Path("reports/Latex_report/tables")
OUT = Path("research_paper/Latex/tables")
MODELS = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16", "convnext_tiny": "ConvNeXt-T"}
INITS = {"pretrained": "ImageNet-pretrained", "scratch": "Random initialization"}
MARKS = {"gain": r"$\uparrow$", "loss": r"$\downarrow$", "inconclusive": r"$\sim$"}
BUDGET_ORDER = ["2", "5", "10", "20", "50", "full"]


def read(name: str) -> list[dict]:
    with open(SOURCE / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell(row: dict | None) -> str:
    """Mean change with its SD and verdict, one decimal; a dagger when the cell has three seeds."""
    if row is None:
        return "--"
    mean, sd = float(row["mean"]), float(row["sd"])
    seeds = len(row["seeds"].split())
    dagger = r"$^\dagger$" if seeds < 5 else ""
    shown = "0.0" if round(mean, 1) == 0 else f"{mean:+.1f}"
    return rf"${shown}$\,{{\scriptsize$\pm${sd:.1f}}}\,{MARKS[row['verdict']]}{dagger}"


def table(rows: list[dict], columns: list[tuple[str, str]], caption: str, label: str,
          key=lambda r: r["label"]) -> str:
    """One booktabs table: a block per initialization, a row per model and budget, a column per comparison.

    ``columns`` pairs the key of a comparison (``key(row)``) with its column header.
    """
    found = {(r["init"], r["model"], r["budget"], key(r)): r for r in rows}
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{ll" + "c" * len(columns) + "}", r"\toprule",
        "Model & Img./class & " + " & ".join(h for _, h in columns) + r" \\",
    ]
    for init, init_name in INITS.items():
        cells = sorted({(m, b) for (i, m, b, _) in found if i == init},
                       key=lambda mb: (list(MODELS).index(mb[0]), BUDGET_ORDER.index(mb[1])))
        if not cells:
            continue
        lines += [r"\midrule", rf"\multicolumn{{{2 + len(columns)}}}{{l}}{{\textit{{{init_name}}}}} \\"]
        previous = None
        for model, budget in cells:
            name = MODELS[model] if model != previous else ""
            if model != previous and previous is not None:
                lines.append(r"\addlinespace[2pt]")
            previous = model
            values = [cell(found.get((init, model, budget, k))) for k, _ in columns]
            lines.append(" & ".join([name, budget] + values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def two_line(top: str, bottom: str) -> str:
    return rf"\shortstack{{{top}\\{bottom}}}"


def stage1(split: str = "test") -> str:
    """SD against real-only and the three comparisons with RandAugment, both initializations."""
    part = "test_accuracy" if split == "test" else "accuracy"
    rows = []
    for init in INITS:
        for r in read(f"stage1_{init}_{part}_sd_prompt_paired.csv"):
            rows.append({**r, "label": "SD-real"})
        # The vs-RandAugment file stacks one block of cells per column of its report table; a block
        # ends where a (model, budget) cell repeats. The report table's header names the blocks.
        name = f"stage1_{init}_{part}_vs_randaugment"
        header = next(line for line in (SOURCE / f"{name}.tex").read_text(encoding="utf-8").splitlines()
                      if line.startswith("Model &"))
        labels = [h.strip().rstrip("\\").strip().replace("$-$", "-") for h in header.split("&")[2:]]
        blocks, seen = [[]], set()
        for r in read(f"{name}_paired.csv"):
            if (r["model"], r["budget"]) in seen:
                blocks.append([])
                seen = set()
            seen.add((r["model"], r["budget"]))
            blocks[-1].append(r)
        assert len(blocks) == len(labels), f"{name}: {len(blocks)} blocks for columns {labels}"
        for label, block in zip(labels, blocks):
            rows += [{**r, "label": label} for r in block]
    columns = [("SD-real", two_line("SD", r"$-$real")), ("SD-RA", two_line("SD", r"$-$RA")),
               ("LoRA-RA", two_line("LoRA", r"$-$RA")), ("SD+RA-RA", two_line("SD+RA", r"$-$RA"))]
    suffix = "" if split == "test" else "-validation"
    return table(rows, columns,
                 rf"CIFAR-100, Stage 1: change in {split} accuracy from class-prompt (SD) and LoRA images, against "
                 r"real-only and against RandAugment (RA).", f"tab:paper-stage1{suffix}")


def from_study(name: str, study: str, columns: list[tuple[str, str]], caption: str, label: str) -> str:
    rows = [r for r in read(name) if r["study"] == study]
    return table(rows, columns, caption, label)


def ensemble() -> str:
    rows = read("ensemble_paired_test.csv")
    columns = [
        ("(real+SD)$-$(real+real)", two_line("real+SD", r"$-$real+real")),
        ("(real+SDonly)$-$(real+real)", two_line("real+SD-only", r"$-$real+real")),
        ("(RA+SD)$-$(RA+RA)", two_line("RA+SD", r"$-$RA+RA")),
        ("(RA+SDonly)$-$(RA+RA)", two_line("RA+SD-only", r"$-$RA+RA")),
        ("(RA+SD+RA)$-$(RA+RA)", two_line("RA+(SD+RA)", r"$-$RA+RA")),
    ]
    return table(rows, columns,
                 r"CIFAR-100, ensembles: change in test accuracy of an equal-weight ensemble of two classifiers "
                 r"against an ensemble of two real-data classifiers of different seeds. Members are named by "
                 r"their training data: real, RA (real with RandAugment), SD (real with class-prompt images), "
                 r"SD-only (class-prompt images alone), SD+RA (both, with RandAugment).", "tab:paper-ensemble")


def stage3(dataset: str, display: str) -> dict[str, str]:
    rows = read(f"stage3_{dataset}_paired_test.csv")
    out = {}
    conventional = [(f"{s}$-$real", two_line(s, r"$-$real"))
                    for s in ["RA", "TA", "AA", "AugMix", "Mixup", "CutMix"]]
    out["conventional"] = table(
        [r for r in rows if r["study"] == "conventional"], conventional,
        rf"{display}: change in test balanced accuracy from each conventional augmentation against real-only. "
        r"RA: RandAugment, TA: TrivialAugment, AA: AutoAugment (ImageNet policy).",
        f"tab:paper-{dataset}-conventional")
    sd = [("SD$-$real", two_line("SD", r"$-$real")), ("SD$-$RA", two_line("SD", r"$-$RA")),
          ("SD+RA$-$RA", two_line("SD+RA", r"$-$RA")), ("SD 5:1+RA$-$RA", two_line("SD 5:1+RA", r"$-$RA")),
          ("SD 5:1+RA$-$best conv.", two_line("SD 5:1+RA", r"$-$best conv."))]
    out["sd"] = table(
        [r for r in rows if r["study"] == "sd"], sd,
        rf"{display}: change in test balanced accuracy from class-prompt images (SD, 1:1 unless a ratio is "
        r"given) against their controls. Best conv.: the conventional augmentation with the highest mean "
        r"validation balanced accuracy in that row.", f"tab:paper-{dataset}-sd")
    lora = [("LoRA$-$real", two_line("LoRA", r"$-$real")), ("LoRA+RA$-$RA", two_line("LoRA+RA", r"$-$RA")),
            ("LoRA$-$SD", two_line("LoRA", r"$-$SD")),
            ("LoRA+RA$-$best conv.", two_line("LoRA+RA", r"$-$best conv."))]
    out["lora"] = table(
        [r for r in rows if r["study"] == "lora"], lora,
        rf"{display}, LoRA follow-up: change in test balanced accuracy from images of a LoRA adapter fitted to "
        r"each budget's training images (1:1) against their controls.", f"tab:paper-{dataset}-lora")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "stage1": stage1(),
        "ratio": from_study(
            "stage2_paired_test.csv", "ratio",
            [(f"{r}$-${c}", two_line(r, f"$-${c}")) for c in ("real", "RA") for r in ("1:1", "2:1", "5:1")],
            r"CIFAR-100: change in test accuracy from class-prompt images at synthetic:real ratios 1:1, 2:1 and "
            r"5:1, against real-only and against RandAugment (RA).", "tab:paper-ratio"),
        "convnext": from_study(
            "stage2_paired_test.csv", "convnext",
            [("SD$-$real", two_line("SD", r"$-$real")), ("RA$-$real", two_line("RA", r"$-$real")),
             ("SD$-$RA", two_line("SD", r"$-$RA")), ("SD+RA$-$RA", two_line("SD+RA", r"$-$RA"))],
            r"CIFAR-100, ConvNeXt-T: change in test accuracy from class-prompt images (SD) and RandAugment (RA).",
            "tab:paper-convnext"),
        "mixing": from_study(
            "stage2_paired_test.csv", "mixing",
            [("Mixup$-$real", two_line("Mixup", r"$-$real")), ("CutMix$-$real", two_line("CutMix", r"$-$real")),
             ("SD$-$RA", two_line("SD", r"$-$RA")), ("SD$-$Mixup", two_line("SD", r"$-$Mixup")),
             ("SD$-$CutMix", two_line("SD", r"$-$CutMix"))],
            r"CIFAR-100: change in test accuracy from Mixup and CutMix against real-only, and from class-prompt "
            r"images (SD) against RandAugment (RA), Mixup and CutMix.", "tab:paper-mixing"),
        "policies": from_study(
            "augment_paired_test.csv", "policies",
            [(f"{s}$-$real", two_line(s, r"$-$real")) for s in ("RA", "TA", "AA", "AugMix")],
            r"CIFAR-100: change in test accuracy from each conventional augmentation policy against real-only. "
            r"RA: RandAugment, TA: TrivialAugment, AA: AutoAugment (CIFAR-10 policy).", "tab:paper-policies"),
        "stacked": from_study(
            "augment_paired_test.csv", "stacked",
            [("SD+RA$-$RA", two_line("SD+RA", r"$-$RA")), ("SD+TA$-$TA", two_line("SD+TA", r"$-$TA")),
             ("SD+AA$-$AA", two_line("SD+AA", r"$-$AA")),
             ("SD+AugMix$-$AugMix", two_line("SD+AugMix", r"$-$AugMix")),
             ("SD+RA+CutMix$-$RA", two_line("SD+RA+CutMix", r"$-$RA")),
             ("SD 5:1+RA$-$RA", two_line("SD 5:1+RA", r"$-$RA"))],
            r"CIFAR-100: change in test accuracy from adding class-prompt images (SD) on top of each conventional "
            r"policy, against the same policy alone.", "tab:paper-stacked"),
        "ensemble": ensemble(),
    }
    for dataset, display in (("flowers102", "Flowers-102"), ("ham10000", "HAM10000")):
        if not (SOURCE / f"stage3_{dataset}_paired_test.csv").exists():
            print(f"skip {display}: stage3_{dataset}_paired_test.csv is not on this branch yet")
            continue
        for study, text in stage3(dataset, display).items():
            tables[f"{dataset}_{study}"] = text
    for name, text in tables.items():
        path = OUT / f"paper_{name}.tex"
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
