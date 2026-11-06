#!/usr/bin/env bash
# Runs all Stage 2 extras one after another on one GPU (192 runs, about 60 GPU
# hours on the A10G), the ones that matter most for the sign-flip question first:
#   1. ConvNeXt-Tiny learning-rate pilot (6 runs), then the learning rate of each
#      arm is chosen from it (pick_lr.py) and written into train_<init>.json
#   2. ConvNeXt-Tiny, pretrained then scratch (54 runs)
#   3. SD class prompts at 2:1 and 5:1, pretrained then scratch (72 runs)
#   4. Mixup and CutMix, pretrained then scratch (60 runs)
# Every script skips completed runs, so the queue can be restarted at any time;
# a restart keeps the learning rates already chosen. Run from the repository
# root, in tmux:
#   bash src/shellscripts/run_stage2.sh 2>&1 | tee stage2.log
set -euo pipefail

bash src/shellscripts/pilot_convnext.sh
for init in pretrained scratch; do
  python -m src.component.pick_lr --root "runs/pilot-${init}" --model convnext_tiny \
    --config "src/component/configs/train_${init}.json"
done

bash src/shellscripts/run_convnext.sh pretrained
bash src/shellscripts/run_convnext.sh scratch
bash src/shellscripts/run_sd_prompt_ratios.sh pretrained
bash src/shellscripts/run_sd_prompt_ratios.sh scratch
bash src/shellscripts/run_mixup_cutmix.sh pretrained
bash src/shellscripts/run_mixup_cutmix.sh scratch
echo "Stage 2 queue finished"
