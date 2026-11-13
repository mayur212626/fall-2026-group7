#!/usr/bin/env bash
# Stacked augmentation: conventional policies on their own and SD class-prompt
# images on top of them, ResNet-50 and ViT-B/16, seeds 0, 1, 2 by default (the
# results use 0-4: run again with SEEDS="3 4"), with each arm's learning rates
# and step budgets, so every run pairs with the Stage 1 real-only, RandAugment
# and SD runs of the same seed.
#   Conditions: TrivialAugment, AutoAugment (CIFAR-10 policy), AugMix; SD with
#               each of them; SD + CutMix; SD + RandAugment + CutMix; and
#               SD + RandAugment at 5:1 (9 conditions).
#   Pretrained: 5, 20 and 50 images per class (54 runs per seed, 270 for 0-4).
#   Scratch:    5 and 50 images per class (36 runs per seed, 180 for 0-4).
# Runs go to runs/augment-<init>. Completed runs are skipped and an interrupted
# run resumes. Give each GPU one part to share the work:
#   bash src/shellscripts/run_augment.sh pretrained      # ~8 GPU hours per seed
#   bash src/shellscripts/run_augment.sh scratch 5       # ~2.6 GPU hours per seed
#   bash src/shellscripts/run_augment.sh scratch 50      # ~14 GPU hours per seed
# Run from the repository root, in tmux, in the pinned environment.
set -euo pipefail

INIT=${1:-pretrained}
SEEDS=${SEEDS:-0 1 2}
CONFIG=src/component/configs/train_${INIT}.json
ROOT=runs/augment-${INIT}
POOL=data/synthetic/sd15_prompt/pool.npz
if [[ $INIT == pretrained ]]; then BUDGETS=${2:-5 20 50}; else BUDGETS=${2:-5 50}; fi
CONDITIONS="trivialaugment autoaugment augmix sd_prompt_trivialaugment sd_prompt_autoaugment sd_prompt_augmix
            sd_prompt_cutmix sd_prompt_randaugment_cutmix sd_prompt_randaugment:5"
[[ -f $POOL ]] || { echo "missing $POOL (generate_sd_prompt_pool.sh)"; exit 1; }

for seed in $SEEDS; do
  for budget in $BUDGETS; do
    for model in resnet50 vit_b_16; do
      for entry in $CONDITIONS; do
        condition=${entry%%:*}
        ratio=1; [[ $entry == *:* ]] && ratio=${entry##*:}
        name="${INIT}_${model}_${condition}_b${budget}_s${seed}"
        [[ $ratio != 1 ]] && name="${name}_r${ratio}"
        if [[ -f "$ROOT/$name/result.json" ]]; then
          echo "skip $name (done)"
          continue
        fi
        extra=()
        [[ $condition == sd_* ]] && extra=(--synthetic-pool "$POOL" --synthetic-ratio "$ratio")
        echo ">>> $name"
        python -m src.component.train --config "$CONFIG" --model "$model" --init "$INIT" \
          --condition "$condition" --budget "$budget" --seed "$seed" --runs-root "$ROOT" "${extra[@]}"
      done
    done
  done
done
echo "augmentation runs finished: $INIT, budgets $BUDGETS, seeds $SEEDS"
