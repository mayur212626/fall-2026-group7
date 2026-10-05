#!/usr/bin/env bash
# Stage 1, LoRA-adapted Stable Diffusion (sd_lora), for one classifier
# initialization: ResNet-50 and ViT-B/16 at every data budget, training seeds 0, 1, 2, real and
# generated images at 1:1. Each budget uses the pool of its own adapter.
# Learning rates and step budgets are those of that arm's real-only baselines.
# Completed runs are skipped and an interrupted run resumes.
#   bash src/shellscripts/run_sd_lora.sh            # ImageNet-pretrained arm
#   bash src/shellscripts/run_sd_lora.sh scratch    # random-initialization arm
set -euo pipefail

INIT=${1:-pretrained}   # pretrained (first) or scratch (random initialization, run after)
CONFIG=src/component/configs/train_${INIT}.json
ROOT=runs/stage1-${INIT}

for seed in 0 1 2; do
  for budget in 5 10 20 50 full; do
    for model in resnet50 vit_b_16; do
      if [[ -f "$ROOT/${INIT}_${model}_sd_lora_b${budget}_s${seed}/result.json" ]]; then
        echo "skip $model sd_lora b$budget s$seed (done)"
        continue
      fi
      python -m src.component.train --config "$CONFIG" --model "$model" --init "$INIT" \
        --condition sd_lora --budget "$budget" --seed "$seed" --runs-root "$ROOT" \
        --synthetic-pool "data/synthetic/sd15_lora_b${budget}/pool.npz"
    done
  done
done

python -m src.component.summarize_pilot --root "$ROOT"
