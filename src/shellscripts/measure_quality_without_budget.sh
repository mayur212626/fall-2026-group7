#!/usr/bin/env bash
# FID and Improved Precision & Recall of the three generated sets of each budget
# (SD class prompts, the LoRA set, SD img2img), against a real reference that
# leaves out that budget's own training images (pool_quality --exclude-budget).
# The img2img edits and the LoRA adapter are made from those images, so a
# reference that contains them flatters these sets. All three sets of a budget
# use the same reference. Budgets 5 to 50 (the full set has no images left out).
# Results: reports/Latex_report/tables/generator_quality_without_budget.csv; the
# scores against the whole training pool stay in generator_quality.csv. Run from
# the repository root:
#   bash src/shellscripts/measure_quality_without_budget.sh
set -euo pipefail

TABLE=reports/Latex_report/tables/generator_quality_without_budget.csv
for budget in 5 10 20 50; do
  python -m src.component.pool_quality --pool data/synthetic/sd15_prompt/pool.npz --per-class "$budget" \
    --name "sd15_prompt_b${budget}" --exclude-budget "$budget" --table "$TABLE"
  python -m src.component.pool_quality --pool "data/synthetic/sd15_lora_b${budget}/pool.npz" \
    --name "sd15_lora_b${budget}" --exclude-budget "$budget" --table "$TABLE"
  python -m src.component.pool_quality --pool data/synthetic/sd15_img2img/pool.npz --per-class "$budget" \
    --name "sd15_img2img_b${budget}" --exclude-budget "$budget" --table "$TABLE"
done
