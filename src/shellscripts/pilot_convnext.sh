#!/usr/bin/env bash
# Learning-rate pilot for ConvNeXt-Tiny, the Stage 2 third architecture, as for
# the Stage 1 models: real-only at 50 images per class, pilot seed 100 (not a
# training seed), validation data only, at the arm's step budget for that size.
#   Pretrained: AdamW 5e-5, 1e-4, 3e-4 (1,500 steps)
#   Scratch:    AdamW 3e-4, 1e-3, 3e-3 (8,000 steps)
# The learning rate with the highest validation accuracy goes into
# train_<init>.json (pick_lr.py, run by run_stage2.sh); the step
# budgets of the other models are kept. Completed
# runs are skipped and an interrupted run resumes. Run from the repository
# root, in tmux:
#   bash src/shellscripts/pilot_convnext.sh
set -euo pipefail

SEED=100

run() {  # run INIT LR
  local init=$1 lr=$2 root=runs/pilot-$1/lr_$2
  if [[ -f "$root/${init}_convnext_tiny_real_only_b50_s${SEED}/result.json" ]]; then
    echo "skip $init convnext_tiny lr $lr (done)"
    return
  fi
  python -m src.component.train --config "src/component/configs/train_${init}.json" --model convnext_tiny \
    --init "$init" --condition real_only --budget 50 --seed "$SEED" --lr "$lr" --runs-root "$root"
}

for lr in 5e-5 1e-4 3e-4; do run pretrained "$lr"; done
for lr in 3e-4 1e-3 3e-3; do run scratch "$lr"; done

python -m src.component.summarize_pilot --root runs/pilot-pretrained
python -m src.component.summarize_pilot --root runs/pilot-scratch
