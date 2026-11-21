#!/usr/bin/env bash
# Class-conditional StyleGAN2-ADA on the 50-images-per-class training budget
# (5,000 images), then its generated pool (450 images per class, the size of
# the Stable Diffusion pools) for the gan condition. Training resumes from its
# last snapshot if interrupted; a finished run and an existing pool are
# skipped. Run from the repository root, in tmux:
#   KIMG=10000 bash src/shellscripts/train_gan_pool.sh
set -euo pipefail

KIMG=${KIMG:?set KIMG, the thousands of real images shown to the discriminator}
SEED=${SEED:-0}
RUN=runs/gan/stylegan2ada_b50_s${SEED}
POOL_DIR=data/synthetic/stylegan2ada_b50

if [[ ! -f "$RUN/result.json" ]]; then
  python -m src.component.train_gan --budget 50 --seed "$SEED" --kimg "$KIMG" --snapshot-kimg 200
fi
if [[ ! -f "$POOL_DIR/pool.npz" ]]; then
  python -m src.component.generate_gan --run-dir "$RUN" --output-dir "$POOL_DIR"
fi
