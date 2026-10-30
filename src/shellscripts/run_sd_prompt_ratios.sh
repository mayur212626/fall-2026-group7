#!/usr/bin/env bash
# Stage 2 synthetic:real ratios 2:1 and 5:1 for SD class prompts (the 1:1 runs
# are Stage 1's sd_prompt). ResNet-50 and ViT-B/16, seeds 0, 1, 2, with the
# arm's learning rates and step budgets, so every run pairs with its real-only
# and 1:1 runs of the same seed. The prompt pool has 450 images per class, so
# 5:1 is possible up to 50 real images per class.
#   Pretrained: 5, 10, 20 and 50 images per class (48 runs).
#   Scratch:    5 and 50 images per class (24 runs).
# Each ratio has its own folder, runs/stage2-ratio-<init>/r<ratio>, because the
# condition name is the same at every ratio. Completed runs are skipped and
# an interrupted run resumes. Run from the repository root, in tmux:
#   bash src/shellscripts/run_sd_prompt_ratios.sh            # ImageNet-pretrained arm
#   bash src/shellscripts/run_sd_prompt_ratios.sh scratch    # random-initialization arm
set -euo pipefail

INIT=${1:-pretrained}
CONFIG=src/component/configs/train_${INIT}.json
POOL=data/synthetic/sd15_prompt/pool.npz
if [[ $INIT == pretrained ]]; then BUDGETS="5 10 20 50"; else BUDGETS="5 50"; fi

for seed in 0 1 2; do
  for ratio in 2 5; do
    root=runs/stage2-ratio-${INIT}/r${ratio}
    for budget in $BUDGETS; do
      for model in resnet50 vit_b_16; do
        if [[ -f "$root/${INIT}_${model}_sd_prompt_b${budget}_s${seed}_r${ratio}/result.json" ]]; then
          echo "skip $model sd_prompt b$budget s$seed r$ratio (done)"
          continue
        fi
        python -m src.component.train --config "$CONFIG" --model "$model" --init "$INIT" \
          --condition sd_prompt --budget "$budget" --seed "$seed" --runs-root "$root" \
          --synthetic-pool "$POOL" --synthetic-ratio "$ratio"
      done
    done
  done
done
