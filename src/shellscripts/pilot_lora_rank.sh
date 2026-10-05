#!/usr/bin/env bash
# LoRA rank pilot (proposal, Phase 5: ranks 4-16), before the main adapters.
# Validation data only, pilot seed 100 (not a main training seed), 50 images
# per class, as in the classifier pilots. For ranks 4, 8 and 16: fit an adapter,
# generate a 1:1 pool (50 per class) and train ResNet-50 and ViT-B/16 on real +
# generated images. choose_lora_rank then writes data/lora/chosen_rank.json,
# which train_lora_adapters.sh uses. About 4-5 GPU hours. Finished steps are
# skipped. Run from the repository root:
#   bash src/shellscripts/pilot_lora_rank.sh
set -euo pipefail

CONFIG=src/component/configs/train_pretrained.json
for rank in 4 8 16; do
  name="sd15_lora_b50_r${rank}"
  python -m src.component.train_lora --budget 50 --rank "$rank" --output-dir "data/lora/pilot/${name}"
  python -m src.component.generate_sd --lora-dir "data/lora/pilot/${name}" \
    --output-dir "data/synthetic/pilot/${name}" --per-class 50 --generator-id "pilot_${name}"
  for model in resnet50 vit_b_16; do
    root="runs/pilot-lora/r${rank}"
    if [[ -f "$root/pretrained_${model}_sd_lora_b50_s100/result.json" ]]; then
      echo "skip $model rank $rank (done)"
      continue
    fi
    python -m src.component.train --config "$CONFIG" --model "$model" --init pretrained \
      --condition sd_lora --budget 50 --seed 100 --runs-root "$root" \
      --synthetic-pool "data/synthetic/pilot/${name}/pool.npz"
  done
done

python -m src.component.choose_lora_rank --root runs/pilot-lora --output data/lora/chosen_rank.json
