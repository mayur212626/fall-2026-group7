#!/usr/bin/env bash
# Stage 1 baselines for the ImageNet-pretrained arm: real-only and RandAugment
# for ResNet-50 and ViT-B/16 at every data budget, training seeds 0, 1, 2.
# Learning rates and step budgets come from train_pretrained.json. Each seed
# finishes completely before the next one starts. Completed runs are skipped
# and an interrupted run resumes, so the script can be started again at any
# time. Run from the repository root, in tmux:
#   bash src/shellscripts/run_pretrained_baselines.sh
set -euo pipefail

CONFIG=src/component/configs/train_pretrained.json
ROOT=runs/stage1-pretrained

for seed in 0 1 2; do
  for budget in 5 10 20 50 full; do
    for model in resnet50 vit_b_16; do
      for condition in real_only randaugment; do
        if [[ -f "$ROOT/pretrained_${model}_${condition}_b${budget}_s${seed}/result.json" ]]; then
          echo "skip $model $condition b$budget s$seed (done)"
          continue
        fi
        python -m src.component.train --config "$CONFIG" --model "$model" --init pretrained \
          --condition "$condition" --budget "$budget" --seed "$seed" --runs-root "$ROOT"
      done
    done
  done
done

python -m src.component.summarize_pilot --root "$ROOT"
