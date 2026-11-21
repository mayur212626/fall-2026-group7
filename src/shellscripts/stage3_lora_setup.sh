#!/usr/bin/env bash
# Stage 3 LoRA follow-up (added after the Stage 3 test results), setup for one
# dataset, on one instance. For every data budget of the dataset:
#   1. a LoRA adapter on that budget's real training images only, with the
#      CIFAR-100 settings unchanged (rank 8, lr 1e-4, batch 8, the CIFAR step
#      count of the same images-per-class budget);
#   2. a generated pool from that adapter, as many images per class as the
#      budget has real ones (1:1);
#   3. the memorization check: how close each generated image is to its
#      nearest training image, against held-out validation images.
# Then the real-versus-generated sample grids. About 1 GPU hour for Flowers-102
# and 2 for HAM10000. Finished adapters and pools are skipped.
# Run from the repository root, in tmux, in the pinned environment:
#   bash src/shellscripts/stage3_lora_setup.sh flowers102
set -euo pipefail

DATASET=${1:?usage: stage3_lora_setup.sh flowers102|ham10000}
BUDGETS=$(python -c "from src.component.stage3_data import DATASETS; print(' '.join(DATASETS['$DATASET']['budgets']))")

for budget in $BUDGETS; do
  name="sd15_lora_${DATASET}_b${budget}"
  python -m src.component.train_lora --dataset "$DATASET" --budget "$budget" --rank 8 --output-dir "data/lora/$name"
  python -m src.component.generate_sd --dataset "$DATASET" --lora-dir "data/lora/$name" \
    --output-dir "data/synthetic/$name" --per-class "$budget"
  python -m src.component.pool_memorization --dataset "$DATASET" --budget "$budget" \
    --pool "data/synthetic/$name/pool.npz" --figure "reports/Latex_report/fig/${name}_nearest"
done

for budget in $(echo "$BUDGETS" | awk '{print $1, $NF}'); do
  name="sd15_lora_${DATASET}_b${budget}"
  python -m src.component.sample_grid --dataset "$DATASET" --pool "data/synthetic/$name/pool.npz" \
    --output "reports/Latex_report/fig/${name}_samples" --label "SD 1.5 + LoRA (${budget} real images per class)" \
    --per-side "$([ "$budget" -lt 5 ] && echo "$budget" || echo 5)"
done
echo "LoRA setup done for $DATASET: check the memorization lines above and the figures in reports/Latex_report/fig/"
