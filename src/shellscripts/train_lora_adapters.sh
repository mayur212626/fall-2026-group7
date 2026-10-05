#!/usr/bin/env bash
# One Stable Diffusion LoRA adapter per data budget, each fitted only on that
# budget's real training images (sd_lora condition), at the rank chosen by the
# rank pilot (data/lora/chosen_rank.json; rank 8 if the pilot has not run).
# About 4-5 GPU hours for all five on the A10G. Budgets that already have an
# adapter are skipped. Run from the repository root:
#   bash src/shellscripts/train_lora_adapters.sh
set -euo pipefail

RANK=$(python -c "import json,pathlib; p=pathlib.Path('data/lora/chosen_rank.json'); print(json.loads(p.read_text())['chosen_rank'] if p.exists() else 8)")
echo "LoRA rank $RANK"
for budget in 5 10 20 50 full; do
  python -m src.component.train_lora --budget "$budget" --rank "$RANK" --output-dir "data/lora/sd15_lora_b${budget}"
done
