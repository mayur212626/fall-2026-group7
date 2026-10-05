#!/usr/bin/env bash
# One generated pool per LoRA adapter, with as many images per class as the
# budget has real images (1:1): 5, 10, 20, 50 and 450 per class, 53,500
# images in total. Finished classes are skipped, so the script can be started
# again. Run from the repository root, after train_lora_adapters.sh:
#   bash src/shellscripts/generate_sd_lora_pools.sh
set -euo pipefail

declare -A PER_CLASS=([5]=5 [10]=10 [20]=20 [50]=50 [full]=450)
for budget in 5 10 20 50 full; do
  name="sd15_lora_b${budget}"
  python -m src.component.generate_sd --lora-dir "data/lora/${name}" --output-dir "data/synthetic/${name}" \
    --per-class "${PER_CLASS[$budget]}" --generator-id "lora_sd_b${budget}"
done

for budget in 5 50; do
  python -m src.component.sample_grid --pool "data/synthetic/sd15_lora_b${budget}/pool.npz" \
    --output "reports/Latex_report/fig/sd15_lora_b${budget}_samples" \
    --label "SD 1.5 + LoRA (${budget} real images per class)"
done
