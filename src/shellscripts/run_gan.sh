#!/usr/bin/env bash
# Class-conditional StyleGAN2-ADA images (gan), for one classifier
# initialization: ResNet-50 and ViT-B/16 at budget 50 (the budget the GAN was
# trained on), training seeds 0 to 4, real and generated images at 1:1. Learning
# rates and step budgets are those of that arm's real-only baselines, so every
# run pairs with its baseline of the same seed. Needs the pool from
# train_gan_pool.sh. Completed runs are skipped and an interrupted run resumes.
# Run from the repository root, in tmux:
#   bash src/shellscripts/run_gan.sh            # ImageNet-pretrained arm
#   bash src/shellscripts/run_gan.sh scratch    # random-initialization arm
set -euo pipefail

INIT=${1:-pretrained}   # pretrained (first) or scratch (random initialization, run after)
CONFIG=src/component/configs/train_${INIT}.json
ROOT=runs/gan-${INIT}
POOL=data/synthetic/stylegan2ada_b50/pool.npz

for seed in 0 1 2 3 4; do
  for model in resnet50 vit_b_16; do
    if [[ -f "$ROOT/${INIT}_${model}_gan_b50_s${seed}/result.json" ]]; then
      echo "skip $model gan b50 s$seed (done)"
      continue
    fi
    python -m src.component.train --config "$CONFIG" --model "$model" --init "$INIT" \
      --condition gan --budget 50 --seed "$seed" --runs-root "$ROOT" \
      --synthetic-pool "$POOL"
  done
done
