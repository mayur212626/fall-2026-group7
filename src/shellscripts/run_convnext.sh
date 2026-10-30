#!/usr/bin/env bash
# Stage 2 third architecture, ConvNeXt-Tiny, with the learning rate chosen by
# pilot_convnext.sh and written into train_<init>.json by pick_lr.py.
# Seeds 0, 1, 2.
#   Pretrained: real-only, RandAugment, SD class prompts and SD + RandAugment,
#               at 5, 20 and the full set of images per class (36 runs).
#   Scratch:    real-only, RandAugment and SD class prompts at 5 and 50 images
#               per class (18 runs): enough to test whether SD loses to
#               RandAugment from random weights, as for the Stage 1 models.
#               The full set is left out: 50,000 steps per run from scratch.
# Runs go to runs/stage2-convnext-<init>. Completed runs are skipped and an
# interrupted run resumes. Run from the repository root, in tmux:
#   bash src/shellscripts/run_convnext.sh            # ImageNet-pretrained arm
#   bash src/shellscripts/run_convnext.sh scratch    # random-initialization arm
set -euo pipefail

INIT=${1:-pretrained}
CONFIG=src/component/configs/train_${INIT}.json
ROOT=runs/stage2-convnext-${INIT}
POOL=data/synthetic/sd15_prompt/pool.npz
if [[ $INIT == pretrained ]]; then
  BUDGETS="5 20 full"; CONDITIONS="real_only randaugment sd_prompt sd_prompt_randaugment"
else
  BUDGETS="5 50"; CONDITIONS="real_only randaugment sd_prompt"
fi

for seed in 0 1 2; do
  for budget in $BUDGETS; do
    for condition in $CONDITIONS; do
      if [[ -f "$ROOT/${INIT}_convnext_tiny_${condition}_b${budget}_s${seed}/result.json" ]]; then
        echo "skip convnext_tiny $condition b$budget s$seed (done)"
        continue
      fi
      pool=()
      [[ $condition == sd_* ]] && pool=(--synthetic-pool "$POOL")
      python -m src.component.train --config "$CONFIG" --model convnext_tiny --init "$INIT" \
        --condition "$condition" --budget "$budget" --seed "$seed" --runs-root "$ROOT" "${pool[@]}"
    done
  done
done
