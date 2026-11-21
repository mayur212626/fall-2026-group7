#!/usr/bin/env bash
# Stage 3 setup for one dataset (flowers102 or ham10000), on one instance:
#   Step 1: download (HAM10000 from Harvard Dataverse; Flowers-102 downloads in step 2).
#   Step 2: store the images at 224x224 and write the split manifest
#           src/component/configs/splits/<dataset>.json (commit it).
#   Step 3: Stable Diffusion 1.5 class-prompt pool: a quick check of 2 classes, then
#           the full pool (Flowers-102 50 and HAM10000 250 images per class), resumable.
#   Step 4: the real-versus-generated sample grid. LOOK AT IT before the full runs.
# Run from the repository root, in tmux, in the pinned environment:
#   bash src/shellscripts/stage3_setup.sh ham10000
# If the HAM10000 download fails, download HAM10000_images_part_1.zip,
# HAM10000_images_part_2.zip and HAM10000_metadata from
# https://doi.org/10.7910/DVN/DBW86T into data/ham10000/raw/ and run the script again.
set -euo pipefail

DATASET=${1:?usage: stage3_setup.sh flowers102|ham10000}
POOL_DIR=data/synthetic/sd15_prompt_${DATASET}

if [[ $DATASET == ham10000 && ! -f data/ham10000/images.npy ]]; then
  python -m src.component.stage3_data download --dataset ham10000
fi
if [[ ! -f data/$DATASET/images.npy ]]; then
  python -m src.component.stage3_data prepare --dataset "$DATASET"
fi

python -m src.component.generate_sd --dataset "$DATASET" --output-dir "data/synthetic/check/sd15_prompt_${DATASET}" \
  --classes 0 1 --per-class 8 --preview 4
python -m src.component.generate_sd --dataset "$DATASET" --output-dir "$POOL_DIR"

python -m src.component.sample_grid --dataset "$DATASET" --pool "$POOL_DIR/pool.npz" \
  --output "reports/Latex_report/fig/${DATASET}_sd15_prompt_samples" --label "Stable Diffusion 1.5, class prompts"
echo "setup done: look at reports/Latex_report/fig/${DATASET}_sd15_prompt_samples.png and $POOL_DIR/preview/"
