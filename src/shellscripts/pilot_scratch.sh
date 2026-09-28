#!/usr/bin/env bash
# Pilot runs for the random-initialization (scratch) arm, on validation data
# only, with pilot seed 100 (not one of the main training seeds 0, 1, 2).
#   Phase 1: learning rates at 50 images per class, 8,000 steps (about 205
#            epochs): SGD 0.03, 0.1, 0.3, 1.0 for ResNet-50 and AdamW 1e-4,
#            3e-4, 1e-3 for ViT-B/16.
#   Phase 2: step budgets for the other data sizes at the chosen learning
#            rate; ViT-B/16 at 3e-4.
# Learning rates are passed explicitly, so each run keeps the settings it
# was trained with. Completed runs are skipped and an interrupted run
# resumes, so the script can be started again at any time. Run from the
# repository root, in tmux:
#   bash src/shellscripts/pilot_scratch.sh
set -euo pipefail

CONFIG=src/component/configs/train_scratch.json
ROOT=runs/pilot-scratch
SEED=100
declare -A STEPS=([5]=2000 [10]=3000 [20]=4000 [50]=8000 [full]=50000)

run() {  # run MODEL BUDGET RUNS_ROOT [more train.py options]
  local model=$1 budget=$2 root=$3
  shift 3
  if [[ -f "$root/scratch_${model}_real_only_b${budget}_s${SEED}/result.json" ]]; then
    echo "skip $root $model b$budget (done)"
    return
  fi
  python -m src.component.train --config "$CONFIG" --model "$model" --init scratch \
    --condition real_only --budget "$budget" --seed "$SEED" --steps "${STEPS[$budget]}" \
    --runs-root "$root" "$@"
}

for lr in 0.03 0.1 0.3 1.0; do run resnet50 50 "$ROOT/lr_$lr" --lr "$lr"; done
for lr in 1e-4 3e-4 1e-3; do run vit_b_16 50 "$ROOT/lr_$lr" --lr "$lr"; done

for budget in 5 10 20 full; do run vit_b_16 "$budget" "$ROOT/lr_3e-4" --lr 3e-4; done

python -m src.component.summarize_pilot --root "$ROOT"
