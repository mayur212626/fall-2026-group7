#!/usr/bin/env bash
# Stage 1, pretrained Stable Diffusion with class prompts (sd_prompt):
# ResNet-50 and ViT-B/16 at every data budget, training seeds 0, 1, 2, real and
# generated images at 1:1, for one classifier initialization. Learning rates
# and step budgets are those of that arm's real-only baselines
# (train_<init>.json), so every run pairs with its baseline of the same seed. Needs the pool from
# generate_sd_prompt_pool.sh. Completed runs are skipped and an interrupted
# run resumes. Run from the repository root, in tmux:
#   bash src/shellscripts/run_sd_prompt.sh            # ImageNet-pretrained arm
#   bash src/shellscripts/run_sd_prompt.sh scratch    # random-initialization arm
set -euo pipefail

INIT=${1:-pretrained}   # pretrained (first) or scratch (random initialization, run after)
CONFIG=src/component/configs/train_${INIT}.json
ROOT=runs/stage1-${INIT}
POOL=data/synthetic/sd15_prompt/pool.npz

for seed in 0 1 2; do
  for budget in 5 10 20 50 full; do
    for model in resnet50 vit_b_16; do
      if [[ -f "$ROOT/${INIT}_${model}_sd_prompt_b${budget}_s${seed}/result.json" ]]; then
        echo "skip $model sd_prompt b$budget s$seed (done)"
        continue
      fi
      python -m src.component.train --config "$CONFIG" --model "$model" --init "$INIT" \
        --condition sd_prompt --budget "$budget" --seed "$seed" --runs-root "$ROOT" \
        --synthetic-pool "$POOL"
    done
  done
done

python -m src.component.summarize_pilot --root "$ROOT"
