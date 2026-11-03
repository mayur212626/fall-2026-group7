#!/usr/bin/env bash
# Stable Diffusion img2img pool (sd_img2img): one edit of each of the first 50
# training images per class, in budget order, so budgets 5 to 50 each use edits
# of their own real images. 5,000 images; resumable (finished classes are
# skipped). Then FID and Improved Precision & Recall at each budget's size.
# Run from the repository root, in tmux:
#   bash src/shellscripts/generate_sd_img2img_pool.sh
set -euo pipefail

OUT=data/synthetic/sd15_img2img
python -m src.component.generate_img2img --output-dir "$OUT"

for budget in 5 10 20 50; do
  registry=()
  [[ $budget == 50 ]] && registry=(--generator sd_img2img)
  python -m src.component.pool_quality --pool "$OUT/pool.npz" --per-class "$budget" \
    --name "sd15_img2img_b${budget}" "${registry[@]}"
done
