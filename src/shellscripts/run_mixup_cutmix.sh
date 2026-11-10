#!/usr/bin/env bash
# Stage 2 Mixup and CutMix (src/component/mixing.py) on a reduced grid that
# serves the sign-flip question: ResNet-50 and ViT-B/16, seeds 0, 1, 2.
#   Pretrained: 5, 20 and the full set of images per class (36 runs).
#   Scratch:    5 and 50 images per class (24 runs); the full set is left out
#               (50,000 steps per run from scratch).
# Learning rates and step budgets are those of the arm's real-only runs, so
# each run pairs with real-only, RandAugment and SD of the same seed. Runs go
# to runs/stage2-mixing-<init>, apart from the Stage 1 runs. Completed runs
# are skipped and an interrupted run resumes. Run from the repository root:
#   bash src/shellscripts/run_mixup_cutmix.sh            # pretrained arm
#   bash src/shellscripts/run_mixup_cutmix.sh scratch    # random-initialization arm
set -euo pipefail

INIT=${1:-pretrained}
CONFIG=src/component/configs/train_${INIT}.json
ROOT=runs/stage2-mixing-${INIT}
if [[ $INIT == pretrained ]]; then BUDGETS="5 20 full"; else BUDGETS="5 50"; fi

for seed in 0 1 2; do
  for budget in $BUDGETS; do
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
