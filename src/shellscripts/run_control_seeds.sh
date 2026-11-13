#!/usr/bin/env bash
# Training seeds 3 and 4 for the Stage 1 controls that the Stage 2 comparisons
# (Mixup, CutMix, synthetic:real ratios) are paired with and that do not have
# them yet: real-only, RandAugment and SD class prompts for ResNet-50 and
# ViT-B/16, ImageNet-pretrained at 50 images per class and the full set, and
# randomly initialized at 5 and 50 images per class (48 runs). The pretrained
# cells at 5, 10 and 20 images per class already have these seeds from
# run_extra_seeds.sh. Settings are those of the seed 0-2 runs (the same configs
# and sd_prompt pool), so only the training seed changes. Completed runs are
# skipped and an interrupted run resumes. Run from the repository root, in tmux:
#   bash src/shellscripts/run_control_seeds.sh
set -euo pipefail

POOL=data/synthetic/sd15_prompt/pool.npz
declare -A BUDGETS=([pretrained]="50 full" [scratch]="5 50")

for init in pretrained scratch; do
  CONFIG=src/component/configs/train_${init}.json
  ROOT=runs/stage1-${init}
  for condition in real_only randaugment sd_prompt; do
    pool_args=()
    if [[ $condition == sd_* ]]; then
      if [[ ! -f "$POOL" ]]; then
        echo "skip $condition: $POOL not found"
        continue
      fi
      pool_args=(--synthetic-pool "$POOL")
    fi
    for seed in 3 4; do
      for budget in ${BUDGETS[$init]}; do
        for model in resnet50 vit_b_16; do
          if [[ -f "$ROOT/${init}_${model}_${condition}_b${budget}_s${seed}/result.json" ]]; then
            echo "skip $init $model $condition b$budget s$seed (done)"
            continue
          fi
          python -m src.component.train --config "$CONFIG" --model "$model" --init "$init" \
            --condition "$condition" --budget "$budget" --seed "$seed" --runs-root "$ROOT" "${pool_args[@]}"
        done
      done
    done
  done
  python -m src.component.summarize_pilot --root "$ROOT"
done
