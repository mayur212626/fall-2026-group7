#!/usr/bin/env bash
# Stage 3 LoRA follow-up runs for one dataset: ResNet-50 and ViT-B/16,
# pretrained and from scratch, every budget, sd_lora and sd_lora_randaugment
# at 1:1, seeds 0-4 by default. Each budget uses the pool of its own adapter
# (stage3_lora_setup.sh). Learning rates and step budgets are the dataset's,
# from the Stage 3 pilots. Runs go to runs/<dataset>-<init>, next to the
# Stage 3 runs they pair with. The runs are split into PARTS lists of about
# equal GPU time; every instance needs the same adapters and pools.
#   bash src/shellscripts/run_stage3_lora.sh flowers102            # one instance
#   bash src/shellscripts/run_stage3_lora.sh ham10000 2 3          # instance 2 of 3
# Completed runs are skipped and an interrupted run resumes. The loop stops if
# the GPU disappears, so no run trains on the CPU.
set -euo pipefail

DATASET=${1:?usage: run_stage3_lora.sh flowers102|ham10000 [PART] [PARTS]}
PART=${2:-1}
PARTS=${3:-1}
SEEDS=${SEEDS:-0 1 2 3 4}
WORKERS=${WORKERS:-6}
python -c "import torch; assert torch.cuda.is_available(), 'no GPU: fix the driver first'"

python -m src.component.stage3_jobs --dataset "$DATASET" --grid lora --seeds $SEEDS --parts "$PARTS" --summary
mapfile -t JOBS < <(python -m src.component.stage3_jobs --dataset "$DATASET" --grid lora --seeds $SEEDS \
  --parts "$PARTS" --part "$PART")
echo "part $PART of $PARTS: ${#JOBS[@]} runs"

for job in "${JOBS[@]}"; do
  read -r init model condition ratio budget seed name <<< "$job"
  root=runs/${DATASET}-${init}
  if [[ -f "$root/$name/result.json" ]]; then
    echo "skip $name (done)"
    continue
  fi
  nvidia-smi > /dev/null || { echo "GPU lost, stopping"; exit 1; }
  pool=data/synthetic/sd15_lora_${DATASET}_b${budget}/pool.npz
  [[ -f $pool ]] || { echo "missing $pool (stage3_lora_setup.sh $DATASET)"; exit 1; }
  echo ">>> $name"
  python -m src.component.train --dataset "$DATASET" --config "src/component/configs/train_${init}_${DATASET}.json" \
    --model "$model" --init "$init" --condition "$condition" --budget "$budget" --seed "$seed" \
    --runs-root "$root" --workers "$WORKERS" --synthetic-pool "$pool" --synthetic-ratio "$ratio" < /dev/null
done
echo "LoRA runs finished: $DATASET part $PART of $PARTS, seeds $SEEDS"
