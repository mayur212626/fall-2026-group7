"""Pool-level RQ3: does a quality measure of a generated set predict how much the set helps?

Each data budget from 5 to 50 images per class has three generated sets: SD
class prompts (``sd_prompt``), the LoRA set of that budget (``sd_lora``) and SD
img2img edits of the budget's own real images (``sd_img2img``); the full set has
the first two. Their FID and Improved Precision & Recall come from
``pool_quality``; their effect on accuracy from the paired tables of
``analyze_results``.

Measures cannot be compared across budgets: both the gain and FID shrink as
the budget grows (FID is biased upwards for small sets), so any measure would
"predict" the gain through the budget alone. The comparison is therefore made
within each budget, model and initialization, for every pair of sets: does the
set the measure calls better also give the larger accuracy gain? A measure that
predicts usefulness agrees in most comparisons; one that does not is at chance.

The gain of each set is its paired change over real-only, averaged over the
training seeds both sets of the pair have. Within a budget the real-only
control is the same for every set, so the ordering would be the same against
RandAugment.

Quality: ``--quality`` takes one or more tables, later ones overriding earlier
ones. By default these are ``generator_quality.csv`` and, when present,
``generator_quality_without_budget.csv``, whose scores at 5 to 50 per class use
a reference without the budget's own images (``pool_quality --exclude-budget``):
the img2img edits and the LoRA adapter are made from those images, and all sets
of a budget are then scored against the same reference. FID minus the
real-versus-real floor is not a separate measure, since the sets of a budget
share the floor.

    python -m src.component.pool_utility --tables reports/Latex_report/tables \
        --figures reports/Latex_report/fig

The comparisons are not independent tests. A measure rates each pair of sets
once per budget, and every model and initialization of the budget shares that
verdict. For a pair whose verdict is the same at every budget, the measure
agrees exactly where the favoured set wins, so those comparisons say how often
that set wins, not whether the measure predicts usefulness. The test therefore
uses only the comparisons of pairs whose verdict changes between budgets (a
two-sided binomial test against one half, Holm over the measures tested); a
measure with no such pair gets no p-value. All comparisons are still counted.

Extra measures (KID, classifier accuracy on the pool, ...) can be added with
``--extra``: a CSV with columns ``generator,budget,<measure>,...``.
"""

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

import numpy as np

GENERATORS = {"sd15_prompt": "sd_prompt", "sd15_lora": "sd_lora", "sd15_img2img": "sd_img2img"}
SETS = ["sd_prompt", "sd_lora", "sd_img2img"]
SET_NAMES = {"sd_prompt": "class prompts", "sd_lora": "LoRA", "sd_img2img": "img2img"}
SET_LETTERS = {"sd_prompt": "P", "sd_lora": "L", "sd_img2img": "I"}
PAIRS = list(combinations(SETS, 2))
BUDGETS = ["5", "10", "20", "50", "full"]
MODEL_NAMES = {"resnet50": "ResNet-50", "vit_b_16": "ViT-B/16"}
# measure -> True when a higher value is better
MEASURES = {"fid": False, "precision": True, "recall": True}
GAIN_FILES = {name: f"stage1_{{init}}_accuracy_{name}_paired.csv" for name in SETS}
PAIR_COLORS = {("sd_prompt", "sd_lora"): "#2a78d6", ("sd_prompt", "sd_img2img"): "#eb6834",
               ("sd_lora", "sd_img2img"): "#1baf7a"}
MARKERS = {"pretrained": "o", "scratch": "^"}
QUALITY_TABLES = ["generator_quality.csv", "generator_quality_without_budget.csv"]


def pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


# --------------------------------------------------------------------------- reading
def read_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _number(value: str | None) -> float:
    return float(value) if value not in (None, "") else float("nan")


def read_quality(paths: list[Path]) -> dict[tuple[str, str], dict[str, float]]:
    """FID, precision and recall of each (set, budget); a later table overrides an earlier one."""
    quality = {}
    for path in paths:
        for row in read_rows(path):
            prefix, _, budget = row["name"].rpartition("_b")
            if prefix not in GENERATORS or budget not in BUDGETS:
                continue
            quality[(GENERATORS[prefix], budget)] = {
                "fid": _number(row["fid"]), "precision": _number(row["precision"]), "recall": _number(row["recall"]),
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
    """One row per (init, model, budget, pair of sets): both gains, both measures, and whether each measure agrees.

    For the pair (a, b), ``gain_difference`` is a's gain minus b's, over the
    seeds both have. ``agree_<measure>`` is 1 when the set the measure calls
    better also has the larger gain, 0 when not, and empty when a value is
    missing or there is a tie.
    """
    rows = []
    for init, gains in gains_by_init.items():
        for model in MODEL_NAMES:
            for budget in BUDGETS:
                for a, b in PAIRS:
                    ga, gb = gains.get((model, a, budget)), gains.get((model, b, budget))
                    qa, qb = quality.get((a, budget)), quality.get((b, budget))
                    if not ga or not gb or qa is None or qb is None:
                        continue
                    seeds = sorted(set(ga) & set(gb))
                    if not seeds:
                        continue
                    per_seed = [ga[s] - gb[s] for s in seeds]
                    diff = float(np.mean(per_seed))
                    row = {"init": init, "model": model, "budget": budget, "pair": pair_name((a, b)),
                           "set_a": a, "set_b": b, "seeds": " ".join(map(str, seeds)),
                           "gain_a": float(np.mean([ga[s] for s in seeds])),
                           "gain_b": float(np.mean([gb[s] for s in seeds])),
                           "gain_difference": diff, "seeds_a_ahead": sum(d > 0 for d in per_seed)}
                    for name, higher in measures.items():
                        va, vb = qa.get(name, float("nan")), qb.get(name, float("nan"))
                        row[f"{name}_a"], row[f"{name}_b"] = va, vb
                        if np.isnan(va) or np.isnan(vb) or va == vb or diff == 0:
                            row[f"agree_{name}"] = ""
                        else:
                            a_better = va > vb if higher else va < vb
                            row[f"agree_{name}"] = int(a_better == (diff > 0))
                    rows.append(row)
    return rows


def budget_verdicts(rows: list[dict], name: str, higher: bool) -> dict[str, dict[str, bool]]:
    """Per pair and budget, whether the measure rates the pair's first set better.

    The verdict depends only on the two sets' values at the budget, so every
    model and initialization of the budget shares it. Budgets where the
    measure cannot say (missing value or tie) are left out.
    """
    verdicts: dict[str, dict[str, bool]] = {}
    for r in rows:
        va, vb = r.get(f"{name}_a", float("nan")), r.get(f"{name}_b", float("nan"))
        if "budget" in r and "pair" in r and not (np.isnan(va) or np.isnan(vb) or va == vb):
            verdicts.setdefault(r["pair"], {})[r["budget"]] = bool(va > vb if higher else va < vb)
    return verdicts


def summary(rows: list[dict], measures: dict[str, bool]) -> list[dict]:
    """Per measure (and per initialization): agreement over all comparisons, and over the informative ones.

    A comparison is informative when its pair's verdict changes between
    budgets. The p-value (binomial against one half) uses only those; Holm is
    taken over the measures that have any.
    """
    out = []
    for name, higher in measures.items():
        verdicts = budget_verdicts(rows, name, higher)
        varying = {pair for pair, v in verdicts.items() if 0 < sum(v.values()) < len(v)}
        for init in [None, *sorted({r["init"] for r in rows})]:
            cells = [r for r in rows if (init is None or r["init"] == init) and r[f"agree_{name}"] != ""]
            informative = [r[f"agree_{name}"] for r in cells if r.get("pair") in varying]
            row = {"measure": name, "init": init or "all", "comparisons": len(cells),
                   "agree": int(sum(r[f"agree_{name}"] for r in cells)),
                   "informative": len(informative), "informative_agree": int(sum(informative)),
                   "varying_pairs": " ".join(sorted(varying)),
                   "p_value": binomial_two_sided(int(sum(informative)), len(informative)) if informative
                   else float("nan")}
            for pair in PAIRS:
                v = verdicts.get(pair_name(pair), {})
                row[f"{pair_name(pair)}_first_better"] = f"{sum(v.values())}/{len(v)}" if v else ""
            out.append(row)
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


def seed_range(seeds: list[str]) -> str:
    """'0 1 2' -> '0--2' for contiguous seeds, else the seeds separated by commas."""
    values = sorted({int(s) for s in seeds})
    if not values:
        return ""
    if values == list(range(values[0], values[-1] + 1)) and len(values) > 1:
        return f"{values[0]}--{values[-1]}"
    return ", ".join(map(str, values))


def summary_table(rows: list[dict], seeds: str = "") -> str:
    """Booktabs table: per measure, its verdict for each pair of sets and its agreement with the gains."""
    inits = sorted({r["init"] for r in rows if r["init"] != "all"})
    cell = {(r["measure"], r["init"]): r for r in rows}
    measures = list(dict.fromkeys(r["measure"] for r in rows))
    names = {"fid": "FID", "precision": "Precision", "recall": "Recall",
             "kid": "KID", "synthetic_accuracy": "Classifier accuracy"}
    seed_text = f", seeds {seeds}" if seeds else ""
    pair_heads = [f"{SET_LETTERS[a]}$>${SET_LETTERS[b]}" for a, b in PAIRS]

    def frac(k: int, n: int) -> str:
        return f"{k}/{n}" if n else "--"

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Does a quality measure pick the more useful generated set? Within each budget, model and "
        r"initialization, every pair of the three sets (P: SD class prompts, L: LoRA, I: img2img; the full set "
        r"has only P and L) is compared: a comparison agrees when the set the measure rates better also gives the "
        r"larger accuracy gain over real-only (paired, mean over the seeds both sets have"
        + seed_text + "). "
        + r"P$>$L, P$>$I and L$>$I give the budgets at which the measure rates the first set of the pair "
        r"better; every model and initialization of a budget shares that verdict. Pretrained, Scratch and All "
        r"count the comparisons in agreement. Where a pair's verdict is "
        r"the same at every budget, the measure agrees exactly where the favoured set wins, so only the "
        r"comparisons of pairs whose verdict changes are tested (Varying): $p$ is a two-sided binomial test "
        r"against one half on those, Holm-adjusted over the measures tested, and -- when there are none.}",
        r"\label{tab:pool-utility}",
        r"\begin{tabular}{l" + "c" * len(PAIRS) + "r" * (len(inits) + 4) + "}",
        r"\toprule",
        " & ".join(["Measure", *pair_heads, *(i.capitalize() for i in inits), "All", "Varying", "$p$",
                    r"Holm $p$"]) + r" \\",
        r"\midrule",
    ]
    for m in measures:
        overall = cell[(m, "all")]
        verdicts = [overall[f"{pair_name(pair)}_first_better"] or "--" for pair in PAIRS]
        parts = [frac(cell[(m, i)]["agree"], cell[(m, i)]["comparisons"]) if (m, i) in cell else "--" for i in inits]
        p = "--" if np.isnan(overall["p_value"]) else f"{overall['p_value']:.3f}"
        ph = "--" if overall["p_holm"] == "" or np.isnan(overall["p_holm"]) else f"{overall['p_holm']:.3f}"
        lines.append(" & ".join([names.get(m, m.replace("_", " ")), *verdicts, *parts,
                                 frac(overall["agree"], overall["comparisons"]),
                                 frac(overall["informative_agree"], overall["informative"]), p, ph]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def plot(rows: list[dict], measures: dict[str, bool], out_stem: Path) -> list[Path]:
    """One panel per measure: for each pair (a, b), b minus a in the measure (x) against in the gain (y)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = list(measures)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(1, len(names), figsize=(2.8 * len(names), 3.0), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        for r in rows:
            x = r[f"{name}_b"] - r[f"{name}_a"]
            if np.isnan(x):
                continue
            pair = (r["set_a"], r["set_b"])
            ax.scatter(x, -r["gain_difference"], s=20, color=PAIR_COLORS[pair], marker=MARKERS[r["init"]],
                       alpha=0.8, linewidths=0)
        ax.axhline(0, color="#8a8a86", linewidth=0.8)
        ax.axvline(0, color="#8a8a86", linewidth=0.8)
        better = "higher" if measures[name] else "lower"
        ax.set_title(f"{name.replace('_', ' ')} ({better} is better)", fontsize=9)
        ax.set_xlabel("second $-$ first set, measure")
        ax.grid(True, color="#e0e0e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("second $-$ first set, gain (pp)")
    handles = [Line2D([], [], color=PAIR_COLORS[p], marker="s", linestyle="",
                      label=f"{SET_NAMES[p[1]]} vs {SET_NAMES[p[0]]}") for p in PAIRS]
    handles += [Line2D([], [], color="#5f5e5a", marker=MARKERS[i], linestyle="", label=i) for i in MARKERS]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
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
    parser.add_argument("--quality", type=Path, nargs="+",
                        help="quality tables, later ones override earlier ones (default: " + ", ".join(QUALITY_TABLES)
                        + " in --tables, the second when present)")
    parser.add_argument("--extra", type=Path, help="CSV with generator,budget and extra measure columns")
    parser.add_argument("--higher-is-better", nargs="*", default=["synthetic_accuracy"])
    parser.add_argument("--lower-is-better", nargs="*", default=["kid"])
    args = parser.parse_args()

    quality_paths = args.quality or [p for p in (args.tables / n for n in QUALITY_TABLES) if p.exists()]
    quality = read_quality(quality_paths)
    measures = dict(MEASURES)
    if args.extra:
        directions = {**{m: True for m in args.higher_is_better}, **{m: False for m in args.lower_is_better}}
        for name in add_extra(quality, args.extra, directions):
            measures[name] = directions[name]
    gains = {init: read_gains(args.tables, init) for init in ("pretrained", "scratch")}
    gains = {init: g for init, g in gains.items() if g}
    rows = comparisons(quality, gains, measures)
    if not rows:
        raise SystemExit("no comparisons: check the paired tables and the quality tables")
    stats = summary(rows, measures)
    print("quality from " + ", ".join(map(str, quality_paths)))
    print(f"wrote {write_rows(rows, args.tables / 'pool_utility_comparisons.csv')}")
    print(f"wrote {write_rows(stats, args.tables / 'pool_utility_summary.csv')}")
    seeds = seed_range([s for r in rows for s in r["seeds"].split()])
    (args.tables / "pool_utility_summary.tex").write_text(summary_table(stats, seeds), encoding="utf-8")
    for path in plot(rows, measures, args.figures / "pool_utility_measure_vs_gain"):
        print(f"wrote {path}")
    print("\nGain difference, first minus second set of each pair (pp), mean over shared seeds:")
    for r in rows:
        print(f"  {r['init']:<10} {MODEL_NAMES[r['model']]:<9} b{r['budget']:<4} {r['pair']:<22} "
              f"{r['gain_difference']:+6.2f}  (first ahead in {r['seeds_a_ahead']}/{len(r['seeds'].split())} seeds)")
    print("\nDoes the measure pick the more useful set? (agree / comparisons; informative = pairs whose verdict "
          "changes between budgets)")
    for s in stats:
        verdicts = "  ".join(f"{pair_name(p)} {s[f'{pair_name(p)}_first_better'] or '--'}" for p in PAIRS)
        p = "p    --" if np.isnan(s["p_value"]) else f"p {s['p_value']:.4f}"
        holm_p = "" if s["p_holm"] == "" or np.isnan(s["p_holm"]) else f"  Holm {s['p_holm']:.4f}"
        print(f"  {s['measure']:<12} {s['init']:<10} all {s['agree']:>2}/{s['comparisons']:<2}  informative "
              f"{s['informative_agree']:>2}/{s['informative']:<2}  {p}{holm_p}   [{verdicts}]")


if __name__ == "__main__":
    main()
