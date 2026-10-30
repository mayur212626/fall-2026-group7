#!/usr/bin/env bash
# Train on generated images only (sd_prompt_only): the SD class-prompt pool
# replaces the real images, with as many generated images per class as the
# real budget has (5, 10, 20, 50 or 450). It tests why the effect of the
# generated images flips with initialization: a classifier trained on them
# alone is still validated on real images. ResNet-50 and ViT-B/16 at every
# budget, training seeds 0, 1, 2, for one initialization. Learning rates and
# step budgets are those of that arm's real-only runs (train_<init>.json).
# Needs the pool from generate_sd_prompt_pool.sh. Completed runs are skipped
# and an interrupted run resumes. Run from the repository root, in tmux:
#   bash src/shellscripts/run_sd_prompt_only.sh            # ImageNet-pretrained arm
#   bash src/shellscripts/run_sd_prompt_only.sh scratch    # random-initialization arm
set -euo pipefail

INIT=${1:-pretrained}
CONFIG=src/component/configs/train_${INIT}.json
ROOT=runs/stage1-${INIT}
POOL=data/synthetic/sd15_prompt/pool.npz

for seed in 0 1 2; do
  for budget in 5 10 20 50 full; do
    for model in resnet50 vit_b_16; do
      if [[ -f "$ROOT/${INIT}_${model}_sd_prompt_only_b${budget}_s${seed}/result.json" ]]; then
        echo "skip $model sd_prompt_only b$budget s$seed (done)"
        continue
      fi
      python -m src.component.train --config "$CONFIG" --model "$model" --init "$INIT" \
        --condition sd_prompt_only --budget "$budget" --seed "$seed" --runs-root "$ROOT" \
        --synthetic-pool "$POOL"
    done
  done
done

python -m src.component.summarize_pilot --root "$ROOT"
