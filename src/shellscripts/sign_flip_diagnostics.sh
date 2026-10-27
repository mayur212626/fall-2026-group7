#!/usr/bin/env bash
# Sign-flip diagnostics (src/component/sign_flip.py): per-class diversity and
# domain-gap measures of the SD class-prompt pool, the accuracy of real-only
# checkpoints on the generated images, and their correlation with the
# per-class recall changes of error_analysis (5 images per class).
# Needs the pool and a GPU for the features; the checkpoint step runs only
# for real-only runs whose best.pt is on this machine. Run from the
# repository root:
#   bash src/shellscripts/sign_flip_diagnostics.sh
set -euo pipefail

POOL=data/synthetic/sd15_prompt/pool.npz
FEATURES=data/synthetic/sd15_prompt/class_features_50.npz

if [[ -f "$POOL" ]]; then
  [[ -f "$FEATURES" ]] || python -m src.component.sign_flip features --pool "$POOL" --output "$FEATURES"
  python -m src.component.sign_flip class-quality --features "$FEATURES"
else
  echo "skip features: $POOL not found"
fi

runs=()
for init in pretrained scratch; do
  for model in resnet50 vit_b_16; do
    for budget in 5 full; do
      dir="runs/stage1-$init/${init}_${model}_real_only_b${budget}_s0"
      [[ -f "$dir/best.pt" ]] && runs+=("$dir")
    done
  done
done
if (( ${#runs[@]} )) && [[ -f "$POOL" ]]; then
  python -m src.component.sign_flip synthetic-accuracy --pool "$POOL" --runs "${runs[@]}"
else
  echo "skip synthetic-accuracy: no real-only best.pt found"
fi

python -m src.component.sign_flip correlate
