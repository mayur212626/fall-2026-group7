#!/usr/bin/env bash
# Stable Diffusion 1.5 class-prompt pool for the sd_prompt condition: 450
# generated images per CIFAR-100 class (45,000 images), reduced to 32x32.
#   Step 1: two classes with 16 images each, to check the setup and measure
#           the generation speed (a few minutes, written to a separate folder).
#   Step 2: the full pool. Each class is saved when it finishes and finished
#           classes are skipped, so the script can be started again at any
#           time; it merges the pool and adds it to the generator registry.
#   Step 3: the real-versus-generated sample figure for the report.
# Run from the repository root, in tmux, when no training job uses the GPU:
#   bash src/shellscripts/generate_sd_prompt_pool.sh
set -euo pipefail

OUT=data/synthetic/sd15_prompt

python -m src.component.generate_sd --output-dir data/synthetic/check/sd15_prompt \
  --classes 0 3 --per-class 16 --preview 4

python -m src.component.generate_sd --output-dir "$OUT"

python -m src.component.sample_grid --pool "$OUT/pool.npz" \
  --output reports/Latex_report/fig/sd15_prompt_samples
