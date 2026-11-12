#!/usr/bin/env bash
# Stage 3 learning-rate pilots for one dataset, on validation data only, with
# pilot seed 100 (not one of the training seeds 0-4): real-only, at the middle
# budget (Flowers-102 5, HAM10000 20 images per class), three learning rates
# around the CIFAR-100 value per model and initialization (12 runs). pick_lr then
# writes the rate with the best validation balanced accuracy into
# src/component/configs/train_<init>_<dataset>.json (commit both configs).
# About 1.5 GPU hours for Flowers-102 and 3 for HAM10000. Completed runs are
# skipped. Run from the repository root, in tmux:
#   bash src/shellscripts/stage3_pilots.sh ham10000              # both initializations
#   bash src/shellscripts/stage3_pilots.sh ham10000 scratch      # one
set -euo pipefail

DATASET=${1:?usage: stage3_pilots.sh flowers102|ham10000 [pretrained|scratch]}
INITS=${2:-pretrained scratch}
WORKERS=${WORKERS:-6}
SEED=100
BUDGET=$(python -c "from src.component.stage3_data import DATASETS; print(DATASETS['$DATASET']['pilot_budget'])")
declare -A LRS=(
  [pretrained_resnet50]="1e-4 3e-4 1e-3" [pretrained_vit_b_16]="2e-5 5e-5 1e-4"
  [scratch_resnet50]="0.3 1.0 3.0"       [scratch_vit_b_16]="1e-4 3e-4 1e-3"
)

for init in $INITS; do
  CONFIG=src/component/configs/train_${init}_${DATASET}.json
  ROOT=runs/${DATASET}-pilot-${init}
  for model in resnet50 vit_b_16; do
    for lr in ${LRS[${init}_${model}]}; do
      name=${init}_${model}_real_only_b${BUDGET}_s${SEED}
      if [[ -f "$ROOT/lr_$lr/$name/result.json" ]]; then
        echo "skip $ROOT/lr_$lr/$name (done)"
        continue
      fi
      echo ">>> pilot $DATASET $init $model lr $lr"
      python -m src.component.train --dataset "$DATASET" --config "$CONFIG" --model "$model" --init "$init" \
        --condition real_only --budget "$BUDGET" --seed "$SEED" --lr "$lr" --runs-root "$ROOT/lr_$lr" \
        --workers "$WORKERS"
    done
    python -m src.component.pick_lr --root "$ROOT" --model "$model" --budget "$BUDGET" \
      --metric balanced_accuracy --config "$CONFIG"
  done
done
