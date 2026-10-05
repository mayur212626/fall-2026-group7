#!/usr/bin/env bash
# FID and Improved Precision & Recall of every finished generated pool, against
# a class-balanced real reference of the same size from the training pool
# (never the test set). Pools that do not exist yet are skipped, so the script
# can be run after each pool finishes. The class-prompt pool is measured at
# each budget's size (5, 10, 20, 50, 450 per class), so it compares with the
# LoRA pool of the same budget at equal sample counts; its full-size scores
# and each LoRA pool's scores also go to the generator registry. Results:
# reports/Latex_report/tables/generator_quality.csv. The first run downloads
# the Inception weights (about 100 MB). Run from the repository root:
#   bash src/shellscripts/measure_quality.sh
set -euo pipefail

declare -A PER_CLASS=([5]=5 [10]=10 [20]=20 [50]=50 [full]=450)
POOL=data/synthetic/sd15_prompt/pool.npz
for budget in 5 10 20 50 full; do
  if [[ -f "$POOL" ]]; then
    registry=()
    [[ $budget == full ]] && registry=(--generator sd_prompt)
    python -m src.component.pool_quality --pool "$POOL" --per-class "${PER_CLASS[$budget]}" \
      --name "sd15_prompt_b${budget}" "${registry[@]}"
  fi
  LORA="data/synthetic/sd15_lora_b${budget}/pool.npz"
  if [[ -f "$LORA" ]]; then
    python -m src.component.pool_quality --pool "$LORA" --name "sd15_lora_b${budget}" \
      --generator "lora_sd_b${budget}"
  fi
done
python - <<'PY'
import csv
rows = list(csv.DictReader(open("reports/Latex_report/tables/generator_quality.csv", encoding="utf-8")))
cols = ["name", "images", "fid", "fid_sd", "fid_real_floor", "precision", "recall"]
print("  ".join(f"{c:>17}" for c in cols))
for r in rows:
    print("  ".join(f"{r[c]:>17}" for c in cols))
PY
