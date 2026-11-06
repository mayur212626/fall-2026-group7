#!/usr/bin/env bash
# Stage 2 conventional-augmentation conditions: Mixup (alpha 0.2) and CutMix
# (alpha 1.0), every batch, as in the torchvision classification reference
# (src/component/mixing.py). ResNet-50 and ViT-B/16 at every data budget,
# training seeds 0-4, for one classifier initialization. Learning rates and
# step budgets are those of that arm's real-only runs (train_<init>.json), so
# every run pairs with its real-only and RandAugment runs of the same seed.
# Completed runs are skipped and an interrupted run resumes. Run from the
# repository root, in tmux:
#   bash src/shellscripts/run_mixing.sh            # ImageNet-pretrained arm
#   bash src/shellscripts/run_mixing.sh scratch    # random-initialization arm
set -euo pipefail

INIT=${1:-pretrained}
CONFIG=src/component/configs/train_${INIT}.json
ROOT=runs/stage1-${INIT}

for seed in 0 1 2 3 4; do
  for budget in 5 10 20 50 full; do
    for model in resnet50 vit_b_16; do
      for condition in mixup cutmix; do
        if [[ -f "$ROOT/${INIT}_${model}_${condition}_b${budget}_s${seed}/result.json" ]]; then
          echo "skip $model $condition b$budget s$seed (done)"
          continue
        fi
        python -m src.component.train --config "$CONFIG" --model "$model" --init "$INIT" \
          --condition "$condition" --budget "$budget" --seed "$seed" --runs-root "$ROOT"
      done
    done
  done
done

python -m src.component.summarize_pilot --root "$ROOT"
