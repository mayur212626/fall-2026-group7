#!/usr/bin/env bash
# Seeds 3 and 4 for Stage 2 (+ optional Stage 1 controls with CONTROLS=yes).
set -euo pipefail
ARMS=${1:-"pretrained scratch"}
CONTROLS=${CONTROLS:-no}
POOL=data/synthetic/sd15_prompt/pool.npz
[[ -f $POOL ]] || { echo "missing $POOL"; exit 1; }
train() {
  local root=$1 init=$2 model=$3 cond=$4 budget=$5 seed=$6; shift 6
  local name="${init}_${model}_${cond}_b${budget}_s${seed}"
  for a in "$@"; do [[ $a =~ ^[25]$ ]] && name="${name}_r${a}"; done
  if [[ -f "$root/$name/result.json" ]]; then echo "skip $name"; return; fi
  echo ">>> $name"
  python -m src.component.train --config "src/component/configs/train_${init}.json" \
    --model "$model" --init "$init" --condition "$cond" --budget "$budget" --seed "$seed" \
    --runs-root "$root" "$@"
}
for init in $ARMS; do
  if [[ $init == pretrained ]]; then
    CTRL="50 full"; CNX="5 20 full"; CNXC="real_only randaugment sd_prompt sd_prompt_randaugment"; RAT="5 10 20 50"; MIX="5 20 full"
  else
    CTRL="5 50"; CNX="5 50"; CNXC="real_only randaugment sd_prompt"; RAT="5 50"; MIX="5 50"
  fi
  for seed in 3 4; do
    if [[ $CONTROLS == yes ]]; then
      for b in $CTRL; do for m in resnet50 vit_b_16; do
        train runs/stage1-$init $init $m real_only $b $seed
        train runs/stage1-$init $init $m randaugment $b $seed
        train runs/stage1-$init $init $m sd_prompt $b $seed --synthetic-pool $POOL
      done; done
    fi
    for b in $CNX; do for c in $CNXC; do
      if [[ $c == sd_* ]]; then train runs/stage2-convnext-$init $init convnext_tiny $c $b $seed --synthetic-pool $POOL
      else train runs/stage2-convnext-$init $init convnext_tiny $c $b $seed; fi
    done; done
    for r in 2 5; do for b in $RAT; do for m in resnet50 vit_b_16; do
      train runs/stage2-ratio-$init/r$r $init $m sd_prompt $b $seed --synthetic-pool $POOL --synthetic-ratio $r
    done; done; done
    for b in $MIX; do for m in resnet50 vit_b_16; do for c in mixup cutmix; do
      train runs/stage2-mixing-$init $init $m $c $b $seed
    done; done; done
  done
done
echo "seeds 3 and 4 done for: $ARMS"
