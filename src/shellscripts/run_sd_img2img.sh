#!/usr/bin/env bash
# SD img2img (sd_img2img) runs for one classifier initialization: ResNet-50 and
# ViT-B/16 at 5, 10, 20 and 50 images per class, training seeds 0, 1, 2, real
# and generated images at 1:1 (each budget uses the edits of its own real
# images, the first k per class of the pool). Learning rates and step budgets
# are those of that arm's real-only baselines, so the runs pair with them by
# seed. Completed runs are skipped and an interrupted run resumes.
#   bash src/shellscripts/run_sd_img2img.sh            # ImageNet-pretrained arm
#   bash src/shellscripts/run_sd_img2img.sh scratch    # random-initialization arm
set -euo pipefail

INIT=${1:-pretrained}
CONFIG=src/component/configs/train_${INIT}.json
ROOT=runs/stage1-${INIT}
POOL=data/synthetic/sd15_img2img/pool.npz

for seed in 0 1 2; do
  for budget in 5 10 20 50; do
    for model in resnet50 vit_b_16; do
      if [[ -f "$ROOT/${INIT}_${model}_sd_img2img_b${budget}_s${seed}/result.json" ]]; then
        echo "skip $model sd_img2img b$budget s$seed (done)"
        continue
      fi
      python -m src.component.train --config "$CONFIG" --model "$model" --init "$INIT" \
        --condition sd_img2img --budget "$budget" --seed "$seed" --runs-root "$ROOT" \
        --synthetic-pool "$POOL"
    done
  done
done

python -m src.component.summarize_pilot --root "$ROOT"
