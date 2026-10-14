#!/usr/bin/env bash
# Training seeds 3 and 4 for the Stage 1 cells behind the main claims: the
# ImageNet-pretrained arm, ResNet-50 and ViT-B/16, 5, 10 and 20 images per
# class, for real-only, RandAugment, SD class prompts and SD class prompts +
# RandAugment. With seeds 0-2 these cells have five paired seeds. Settings are
# those of the seed 0-2 runs (train_pretrained.json, the same sd_prompt pool),
# so only the training seed changes. Real-only and RandAugment run first; the
# synthetic conditions need the pool from generate_sd_prompt_pool.sh and are
# skipped with a message while it is missing. Completed runs are skipped and an
# interrupted run resumes. Run from the repository root, in tmux:
#   bash src/shellscripts/run_extra_seeds.sh
set -euo pipefail

CONFIG=src/component/configs/train_pretrained.json
ROOT=runs/stage1-pretrained
POOL=data/synthetic/sd15_prompt/pool.npz

for condition in real_only randaugment sd_prompt sd_prompt_randaugment; do
  pool_args=()
  if [[ $condition == sd_* ]]; then
    if [[ ! -f "$POOL" ]]; then
      echo "skip $condition: $POOL not found"
      continue
    fi
    pool_args=(--synthetic-pool "$POOL")
  fi
  for seed in 3 4; do
    for budget in 5 10 20; do
      for model in resnet50 vit_b_16; do
        if [[ -f "$ROOT/pretrained_${model}_${condition}_b${budget}_s${seed}/result.json" ]]; then
          echo "skip $model $condition b$budget s$seed (done)"
          continue
        fi
        python -m src.component.train --config "$CONFIG" --model "$model" --init pretrained \
          --condition "$condition" --budget "$budget" --seed "$seed" --runs-root "$ROOT" "${pool_args[@]}"
      done
    done
  done
done

python -m src.component.summarize_pilot --root "$ROOT"
