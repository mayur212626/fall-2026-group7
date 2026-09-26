#!/usr/bin/env bash
# Pilot runs for the ImageNet-pretrained arm, on validation data only, with
# pilot seed 100 (not one of the main training seeds 0, 1, 2).
#   Part 1: one long real-only run per model and data budget, to choose the
#           step budget where validation accuracy stops improving.
#   Part 2: two more learning rates per model at 50 images per class; the
#           config's own learning rate is the Part 1 run at that budget.
# Completed runs are skipped and an interrupted run resumes, so the script
# can be started again at any time. Run from the repository root, in tmux:
#   bash src/shellscripts/pilot_pretrained.sh
set -euo pipefail

CONFIG=src/component/configs/train_pretrained.json
ROOT=runs/pilot-pretrained
SEED=100
declare -A STEPS=([5]=600 [10]=800 [20]=1000 [50]=1500 [full]=6000)

run() {  # run MODEL BUDGET RUNS_ROOT [more train.py options]
  local model=$1 budget=$2 root=$3
  shift 3
  if [[ -f "$root/pretrained_${model}_real_only_b${budget}_s${SEED}/result.json" ]]; then
    echo "skip $root $model b$budget (done)"
    return
  fi
  python -m src.component.train --config "$CONFIG" --model "$model" --init pretrained \
    --condition real_only --budget "$budget" --seed "$SEED" --steps "${STEPS[$budget]}" \
    --runs-root "$root" "$@"
}

for budget in 5 10 20 50; do
  for model in resnet50 vit_b_16; do
    run "$model" "$budget" "$ROOT/steps"
  done
done

for lr in 3e-5 3e-4; do run resnet50 50 "$ROOT/lr_$lr" --lr "$lr"; done
for lr in 2e-5 1e-4; do run vit_b_16 50 "$ROOT/lr_$lr" --lr "$lr"; done

for model in resnet50 vit_b_16; do
  run "$model" full "$ROOT/steps"
done

python -m src.component.summarize_pilot --root "$ROOT"
