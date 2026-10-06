#!/usr/bin/env bash
# The Stage 1 diffusion conditions in the proposal's order, in one command.
# Every step skips finished work, so after any interruption the same command
# continues where it stopped. Quick checks of both generators run first, so a
# setup problem shows up in minutes, not after hours.
#
#   Pretrained arm (ImageNet-pretrained classifiers)            ~35 GPU hours
#    1. SD class-prompt check: 16 images of 2 classes           ~5 min
#    2. LoRA check: 300-step adapter + 16 images of 2 classes   ~10 min
#    3. SD class-prompt pool, 45,000 images + sample figure     ~6-8 h
#    4. sd_prompt runs, 30                                      ~6 h
#    5. LoRA rank pilot (ranks 4, 8, 16; seed 100; validation)  ~4-5 h
#    6. LoRA adapters for every budget at the chosen rank       ~4-5 h
#    7. LoRA pools, 53,500 images + sample figures              ~7-9 h
#    8. sd_lora runs, 30                                        ~6 h
#    9. Tables, figure and run matrix
#   Random-initialization arm (needs Mayur's scratch baselines) ~3 GPU days
#   10. sd_prompt and sd_lora runs, 60, with train_scratch.json
#   11. Tables, figure and run matrix
#
# Test scores are left to the one-time evaluate_test step (docs/synthetic-data.md).
# Run from the repository root, in the background:
#   nohup bash src/shellscripts/run_stage1_diffusion.sh > runs/stage1_diffusion.log 2>&1 &
set -euo pipefail
mkdir -p runs

analysis() {  # analysis INIT
  if compgen -G "runs/stage1-$1/*/result.json" > /dev/null; then
    python -m src.component.analyze_results --root "runs/stage1-$1" \
      --figure "reports/Latex_report/fig/stage1_$1_diffusion_accuracy" \
      --table "reports/Latex_report/tables/stage1_$1_diffusion_accuracy"
  fi
  python -m src.component.run_matrix --keep-missing --output src/component/configs/run_matrix.csv
}

echo "== 1. SD class-prompt check"
python -m src.component.generate_sd --output-dir data/synthetic/check/sd15_prompt \
  --classes 0 3 --per-class 16 --preview 4

echo "== 2. LoRA check"
python -m src.component.train_lora --budget 5 --steps 300 --output-dir data/lora/check/sd15_lora_b5
python -m src.component.generate_sd --lora-dir data/lora/check/sd15_lora_b5 \
  --output-dir data/synthetic/check/sd15_lora_b5 --classes 0 3 --per-class 16 --preview 4 \
  --generator-id check_sd15_lora_b5

echo "== 3. SD class-prompt pool"
python -m src.component.generate_sd --output-dir data/synthetic/sd15_prompt
python -m src.component.sample_grid --pool data/synthetic/sd15_prompt/pool.npz \
  --output reports/Latex_report/fig/sd15_prompt_samples

echo "== 4. sd_prompt runs, pretrained arm"
bash src/shellscripts/run_sd_prompt.sh pretrained

echo "== 5. LoRA rank pilot"
bash src/shellscripts/pilot_lora_rank.sh

echo "== 6. LoRA adapters"
bash src/shellscripts/train_lora_adapters.sh

echo "== 7. LoRA pools"
bash src/shellscripts/generate_sd_lora_pools.sh

echo "== 8. sd_lora runs, pretrained arm"
bash src/shellscripts/run_sd_lora.sh pretrained

echo "== 9. Pretrained arm: tables, figure, run matrix"
analysis pretrained

echo "== 10. Random-initialization arm"
bash src/shellscripts/run_sd_prompt.sh scratch
bash src/shellscripts/run_sd_lora.sh scratch

echo "== 11. Random-initialization arm: tables, figure, run matrix"
analysis scratch
echo "== done"
