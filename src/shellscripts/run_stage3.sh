#!/usr/bin/env bash
# Stage 3 training runs for one dataset: ResNet-50 and ViT-B/16, pretrained and
# from scratch, every budget of the dataset, 10 conditions (real-only;
# RandAugment, TrivialAugment, AutoAugment, AugMix, Mixup, CutMix; SD 1:1,
# SD + RandAugment 1:1 and 5:1), seeds 0, 1, 2 by default (then SEEDS="3 4").
# The runs are split into PARTS lists of about equal GPU time, one per instance
# (stage3_jobs); every instance runs the same command with its own PART:
#   bash src/shellscripts/run_stage3.sh ham10000 1 3        # instance 1 of 3
#   bash src/shellscripts/run_stage3.sh ham10000 2 3        # instance 2 of 3
#   SEEDS="3 4" bash src/shellscripts/run_stage3.sh ham10000 1 3
# Runs go to runs/<dataset>-<init>. Completed runs are skipped and an interrupted
# run resumes. Needs the setup (data, manifest, pool) and the pilots (learning
# rates in the configs). Run from the repository root, in tmux.
set -euo pipefail

DATASET=${1:?usage: run_stage3.sh flowers102|ham10000 [PART] [PARTS]}
PART=${2:-1}
PARTS=${3:-1}
SEEDS=${SEEDS:-0 1 2}
WORKERS=${WORKERS:-6}
POOL=data/synthetic/sd15_prompt_${DATASET}/pool.npz
[[ -f $POOL ]] || { echo "missing $POOL (stage3_setup.sh $DATASET)"; exit 1; }
for init in pretrained scratch; do
  found=0
  [[ -d runs/${DATASET}-pilot-${init} ]] && found=$(find "runs/${DATASET}-pilot-${init}" -name result.json | wc -l)
  if [[ $found -lt 6 && -z ${PILOTS_DONE:-} ]]; then
    echo "runs/${DATASET}-pilot-${init} has $found of 6 pilot runs: run stage3_pilots.sh first"
    echo "(or set PILOTS_DONE=1 if the configs already hold the chosen learning rates)"
    exit 1
  fi
done

python -m src.component.stage3_jobs --dataset "$DATASET" --seeds $SEEDS --parts "$PARTS" --summary
mapfile -t JOBS < <(python -m src.component.stage3_jobs --dataset "$DATASET" --seeds $SEEDS --parts "$PARTS" --part "$PART")
echo "part $PART of $PARTS: ${#JOBS[@]} runs"

for job in "${JOBS[@]}"; do
  read -r init model condition ratio budget seed name <<< "$job"
  root=runs/${DATASET}-${init}
  if [[ -f "$root/$name/result.json" ]]; then
    echo "skip $name (done)"
    continue
  fi
  extra=()
  [[ $condition == sd_* ]] && extra=(--synthetic-pool "$POOL" --synthetic-ratio "$ratio")
  echo ">>> $name"
  python -m src.component.train --dataset "$DATASET" --config "src/component/configs/train_${init}_${DATASET}.json" \
    --model "$model" --init "$init" --condition "$condition" --budget "$budget" --seed "$seed" \
    --runs-root "$root" --workers "$WORKERS" "${extra[@]}"
done
echo "stage 3 runs finished: $DATASET part $PART of $PARTS, seeds $SEEDS"
