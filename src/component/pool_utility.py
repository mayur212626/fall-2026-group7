"""Pool-level RQ3: does a quality measure of a generated set predict how much the set helps?

Stage 1 has two generated sets per data budget: SD class prompts (``sd_prompt``)
and the LoRA-adapted pool of that budget (``sd_lora``). Their FID and
Improved Precision & Recall are in ``generator_quality.csv`` (``pool_quality``);
their effect on accuracy is in the paired tables of ``analyze_results``.

Measures cannot be compared across budgets: both the gain and FID shrink as
the budget grows (FID is biased upwards for small sets), so any measure would
"predict" the gain through the budget alone. The comparison is therefore made
within each budget, model and initialization: of the two sets, does the one
the measure calls better also give the larger accuracy gain? A measure that
predicts usefulness agrees in most of these comparisons; one that does not is
at chance (half).

The gain of each set is its paired change over real-only, averaged over the
training seeds both sets have (so the two are compared on the same seeds).
Within a budget the real-only control is the same for both sets, so the
ordering would be the same against RandAugment.

    python -m src.component.pool_utility --tables reports/Latex_report/tables \
        --figures reports/Latex_report/fig

Limitation: with two sets per budget, a measure that rates the same set
better at every budget (e.g. precision always favouring one generator) agrees
exactly where that set wins. The test separates measures that change their
verdict across budgets (FID does) from the gain; separating the rest needs
more generators (e.g. DA-Fusion).

Extra measures (KID, classifier accuracy on the pool, ...) can be added with
``--extra``: a CSV with columns ``generator,budget,<measure>,...``.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

GENERATORS = {"sd15_prompt": "sd_prompt", "sd15_lora": "sd_lora"}
BUDGETS = ["5", "10", "20", "50", "full"]
MODEL_NAMES = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16"}
# measure -> True when a higher value is better
MEASURES = {"fid": False, "fid_gap": False, "precision": True, "recall": True}
GAIN_FILES = {"sd_prompt": "stage1_{init}_accuracy_sd_prompt_paired.csv",
              "sd_lora": "stage1_{init}_accuracy_sd_lora_paired.csv"}
COLORS = {"pretrained": "#2a78d6", "scratch": "#e34948"}


# --------------------------------------------------------------------------- reading
def read_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _number(value: str | None) -> float:
    return float(value) if value not in (None, "") else float("nan")


def read_quality(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Quality of each (generator, budget) from generator_quality.csv; fid_gap is FID minus the real floor."""
    quality = {}
    for row in read_rows(path):
        prefix, _, budget = row["name"].rpartition("_b")
        if prefix not in GENERATORS or budget not in BUDGETS:
            continue
        fid, floor = _number(row["fid"]), _number(row.get("fid_real_floor"))
        quality[(GENERATORS[prefix], budget)] = {
            "fid": fid, "fid_gap": fid - floor, "precision": _number(row["precision"]),
            "recall": _number(row["recall"]),
        }
    return quality


def add_extra(quality: dict[tuple[str, str], dict[str, float]], path: Path,
              higher_is_better: dict[str, bool]) -> list[str]:
    """Merge extra measures (columns besides generator and budget) into ``quality``; return their names."""
    rows = read_rows(path)
    names = [c for c in rows[0] if c not in ("generator", "budget")] if rows else []
    for row in rows:
        quality.setdefault((row["generator"], row["budget"]), {}).update({n: _number(row[n]) for n in names})
    for name in names:
        if name not in higher_is_better:
            raise ValueError(f"say whether a higher {name} is better (--higher-is-better / --lower-is-better)")
    return names


def read_gains(tables: Path, init: str) -> dict[tuple[str, str, str], dict[int, float]]:
    """Per-seed paired change over real-only of each (model, generator, budget)."""
    gains = {}
    for generator, pattern in GAIN_FILES.items():
        path = Path(tables) / pattern.format(init=init)
        if not path.exists():
            continue
        for row in read_rows(path):
            seeds = [int(s) for s in row["seeds"].split()]
            changes = [float(c) for c in row["changes"].split()]
            gains[(row["model"], generator, row["budget"])] = dict(zip(seeds, changes, strict=True))
    return gains


# --------------------------------------------------------------------------- analysis
def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test of k successes in n trials against one half."""
    if n == 0:
        return float("nan")
    tail = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(tail + 1)) / 2 ** n)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN entries are left out of the count and kept as NaN)."""
    p = np.asarray(p_values, dtype=np.float64)
    finite = np.flatnonzero(~np.isnan(p))
    adjusted = np.full(len(p), np.nan)
    running = 0.0
    for position, i in enumerate(finite[np.argsort(p[finite], kind="stable")]):
        running = max(running, min(1.0, (len(finite) - position) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def comparisons(quality: dict, gains_by_init: dict[str, dict], measures: dict[str, bool]) -> list[dict]:
    """One row per (init, model, budget): both sets' gains and measures, and whether each measure agrees.

    ``agree_<measure>`` is 1 when the set the measure calls better also has
    the larger gain, 0 when not, and empty when either value is missing or
    they tie.
    """
    rows = []
    for init, gains in gains_by_init.items():
        for model in MODEL_NAMES:
            for budget in BUDGETS:
                a, b = gains.get((model, "sd_prompt", budget)), gains.get((model, "sd_lora", budget))
                qa, qb = quality.get(("sd_prompt", budget)), quality.get(("sd_lora", budget))
                if not a or not b or qa is None or qb is None:
                    continue
                seeds = sorted(set(a) & set(b))
                if not seeds:
                    continue
                per_seed = [a[s] - b[s] for s in seeds]
                diff = float(np.mean(per_seed))
                row = {"init": init, "model": model, "budget": budget, "seeds": " ".join(map(str, seeds)),
                       "gain_sd_prompt": float(np.mean([a[s] for s in seeds])),
                       "gain_sd_lora": float(np.mean([b[s] for s in seeds])),
                       "gain_difference": diff,
                       "seeds_prompt_ahead": sum(d > 0 for d in per_seed)}
                for name, higher in measures.items():
                    va, vb = qa.get(name, float("nan")), qb.get(name, float("nan"))
                    row[f"{name}_sd_prompt"], row[f"{name}_sd_lora"] = va, vb
                    if np.isnan(va) or np.isnan(vb) or va == vb or diff == 0:
                        row[f"agree_{name}"] = ""
                    else:
                        prompt_better = va > vb if higher else va < vb
                        row[f"agree_{name}"] = int(prompt_better == (diff > 0))
                rows.append(row)
    return rows


def summary(rows: list[dict], measures: dict[str, bool]) -> list[dict]:
    """Per measure (and per initialization): comparisons in agreement, binomial p, Holm over the measures."""
    out = []
    for name in measures:
        for init in [None, *sorted({r["init"] for r in rows})]:
            votes = [r[f"agree_{name}"] for r in rows if (init is None or r["init"] == init)
                     and r[f"agree_{name}"] != ""]
            out.append({"measure": name, "init": init or "all", "comparisons": len(votes),
                        "agree": int(sum(votes)), "agree_share": sum(votes) / len(votes) if votes else float("nan"),
                        "p_value": binomial_two_sided(int(sum(votes)), len(votes))})
    overall = [r for r in out if r["init"] == "all"]
    for row, adjusted in zip(overall, holm([r["p_value"] for r in overall]), strict=True):
        row["p_holm"] = adjusted
    for row in out:
        row.setdefault("p_holm", "")
    return out


# --------------------------------------------------------------------------- output
def write_rows(rows: list[dict], path: Path) -> Path:
    def cell(value: object) -> object:
        return round(value, 4) if isinstance(value, float) else value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: cell(v) for k, v in row.items()} for row in rows)
    return path


def summary_table(rows: list[dict]) -> str:
    """Booktabs table: per measure, comparisons in agreement for each initialization and overall."""
    inits = sorted({r["init"] for r in rows if r["init"] != "all"})
    cell = {(r["measure"], r["init"]): r for r in rows}
    measures = list(dict.fromkeys(r["measure"] for r in rows))
    names = {"fid": "FID", "fid_gap": "FID $-$ real floor", "precision": "Precision", "recall": "Recall",
             "kid": "KID", "synthetic_accuracy": "Classifier accuracy"}

    def frac(r: dict) -> str:
        return f"{r['agree']}/{r['comparisons']}" if r["comparisons"] else "--"

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Does a quality measure pick the more useful generated set? Within each budget, model and "
        r"initialization, the SD class-prompt and LoRA sets are compared: a comparison agrees when the set the "
        r"measure rates better also gives the larger accuracy gain over real-only (paired, mean over the seeds "
        r"both sets have). Chance is one half; $p$ is a two-sided binomial test, Holm-adjusted over the "
        r"measures in the last column. With two sets per budget, a measure that rates the same set better at "
        r"every budget agrees exactly where that set wins, so more generators are needed to separate such "
        r"measures.}",
        r"\label{tab:pool-utility}",
        r"\begin{tabular}{l" + "r" * (len(inits) + 3) + "}",
        r"\toprule",
        "Measure & " + " & ".join(i.capitalize() for i in inits) + r" & All & $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for m in measures:
        overall = cell[(m, "all")]
        parts = [frac(cell[(m, i)]) if (m, i) in cell else "--" for i in inits]
        p = "--" if np.isnan(overall["p_value"]) else f"{overall['p_value']:.3f}"
        ph = "--" if overall["p_holm"] == "" or np.isnan(overall["p_holm"]) else f"{overall['p_holm']:.3f}"
        lines.append(f"{names.get(m, m.replace('_', ' '))} & " + " & ".join(parts) + f" & {frac(overall)} & {p} & {ph}"
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def plot(rows: list[dict], measures: dict[str, bool], out_stem: Path) -> list[Path]:
    """One panel per measure: LoRA minus class prompts in the measure (x) against in the gain (y)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(measures)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(1, len(names), figsize=(2.7 * len(names), 2.8), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        for init, color in COLORS.items():
            pts = [(r[f"{name}_sd_lora"] - r[f"{name}_sd_prompt"], -r["gain_difference"]) for r in rows
                   if r["init"] == init and not np.isnan(r[f"{name}_sd_lora"] - r[f"{name}_sd_prompt"])]
            if pts:
                x, y = zip(*pts, strict=True)
                ax.scatter(x, y, s=16, color=color, label=init, alpha=0.8, linewidths=0)
        ax.axhline(0, color="#8a8a86", linewidth=0.8)
        ax.axvline(0, color="#8a8a86", linewidth=0.8)
        better = "higher" if measures[name] else "lower"
        ax.set_title(f"{name.replace('_', ' ')} ({better} is better)", fontsize=9)
        ax.set_xlabel("LoRA $-$ prompts, measure")
        ax.grid(True, color="#e0e0e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("LoRA $-$ prompts, gain (pp)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [out_stem.with_suffix(".svg"), out_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, default=Path("reports/Latex_report/tables"))
    parser.add_argument("--figures", type=Path, default=Path("reports/Latex_report/fig"))
    parser.add_argument("--quality", type=Path, help="default: <tables>/generator_quality.csv")
    parser.add_argument("--extra", type=Path, help="CSV with generator,budget and extra measure columns")
    parser.add_argument("--higher-is-better", nargs="*", default=["synthetic_accuracy"])
    parser.add_argument("--lower-is-better", nargs="*", default=["kid"])
    args = parser.parse_args()

    quality = read_quality(args.quality or args.tables / "generator_quality.csv")
    measures = dict(MEASURES)
    if args.extra:
        directions = {**{m: True for m in args.higher_is_better}, **{m: False for m in args.lower_is_better}}
        for name in add_extra(quality, args.extra, directions):
            measures[name] = directions[name]
    gains = {init: read_gains(args.tables, init) for init in ("pretrained", "scratch")}
    gains = {init: g for init, g in gains.items() if g}
    rows = comparisons(quality, gains, measures)
    if not rows:
        raise SystemExit("no comparisons: check the paired tables and generator_quality.csv")
    stats = summary(rows, measures)
    print(f"wrote {write_rows(rows, args.tables / 'pool_utility_comparisons.csv')}")
    print(f"wrote {write_rows(stats, args.tables / 'pool_utility_summary.csv')}")
    (args.tables / "pool_utility_summary.tex").write_text(summary_table(stats), encoding="utf-8")
    for path in plot(rows, measures, args.figures / "pool_utility_measure_vs_gain"):
        print(f"wrote {path}")
    print("\nSD class prompts minus LoRA, gain over real-only (pp), mean over shared seeds:")
    for r in rows:
        print(f"  {r['init']:<10} {MODEL_NAMES[r['model']]:<9} b{r['budget']:<4} {r['gain_difference']:+6.2f}  "
              f"(prompts ahead in {r['seeds_prompt_ahead']}/{len(r['seeds'].split())} seeds)")
    print("\nDoes the measure pick the more useful set? (agree / comparisons, binomial p, Holm p)")
    for s in stats:
        holm_p = "" if s["p_holm"] == "" else f"  Holm {s['p_holm']:.4f}"
        print(f"  {s['measure']:<20} {s['init']:<10} {s['agree']:>2}/{s['comparisons']:<2}  p {s['p_value']:.4f}{holm_p}")


if __name__ == "__main__":
    main()
