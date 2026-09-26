# Training runs

`src/component/train.py` trains one run: one model, initialization, condition, data budget and seed. All runs share the same code; the settings come from a config file and the command line.

| Option | Values |
|---|---|
| `--config` | `src/component/configs/train_pretrained.json` or `train_scratch.json` |
| `--model` | `resnet50`, `vit_b_16` |
| `--init` | `pretrained` (ImageNet weights), `scratch` (random initialization) |
| `--condition` | `real_only`, `randaugment` |
| `--budget` | `5`, `10`, `20`, `50`, `full` (images per class, from the split manifest) |
| `--seed` | training seed, e.g. `0`, `1`, `2` |
| `--steps` | overrides the config's step budget; used for pilot runs |
| `--lr` | overrides the config's learning rate; used for pilot runs |

Example, from the repository root:

```bash
python -m src.component.train --config src/component/configs/train_pretrained.json \
    --model resnet50 --init pretrained --condition real_only --budget 50 --seed 0 --steps 500
```

The step budgets in the config files are empty until the pilot runs fix them. Until then, pass `--steps`.

## Pilot runs

`src/shellscripts/pilot_pretrained.sh` fixes the step budgets and confirms the learning rates of the pretrained arm. It uses validation data only and pilot seed 100, which is not one of the main training seeds.

1. One long real-only run per model and data budget (600, 800, 1,000, 1,500 and 6,000 steps for 5, 10, 20, 50 per class and full).
2. Two more learning rates per model at 50 images per class: 3e-5 and 3e-4 for ResNet-50, 2e-5 and 1e-4 for ViT-B/16.

```bash
bash src/shellscripts/pilot_pretrained.sh
```

Finished runs are skipped and an interrupted run resumes, so the script can be started again. At the end it prints a summary; to print it at any time:

```bash
python -m src.component.summarize_pilot --root runs/pilot-pretrained
```

The plateau step is the first validation within 0.5 percentage points of the best accuracy. The step budget for each data size is set from it and written into the config file before the main runs.

## Protocol

- Images are 32×32 CIFAR-100 images, upsampled to 224×224 (bilinear) and normalized with the ImageNet mean and standard deviation.
- `real_only` applies no augmentation. `randaugment` applies RandAugment (2 operations, magnitude 9) to the 32×32 image before upsampling.
- Training runs for a fixed number of optimizer steps: linear warmup over 5% of the steps, then cosine decay; gradient norm clipped at 1.0; label smoothing 0.1; bf16 mixed precision on the GPU.
- The validation set is scored 20 times at equal step intervals. The best checkpoint has the highest validation accuracy; ties go to the lower validation loss, then the earlier step.
- Seeds, cuDNN determinism and a fixed sample order make each run reproducible. The order depends only on the seed and the step, so a resumed run sees the same batches and augmentations as an uninterrupted one.
- The official test set is not used by the runner.

## Output

Each run writes to `runs/<init>_<model>_<condition>_b<budget>_s<seed>/`:

| File | Content |
|---|---|
| `config.json` | Settings, package versions, GPU, weight version, manifest hash, git commit |
| `history.jsonl` | Per validation: step, training loss, learning rate, validation accuracy, macro-F1, balanced accuracy, loss, elapsed time |
| `best.pt` | Weights of the best validation checkpoint |
| `result.json` | Best validation scores, per-class recall, confusion matrix, validation predictions with image IDs, training time and peak GPU memory |

`last.pt` holds the full training state while the run is in progress and is removed when it completes. If a run is interrupted, run the same command again to resume. The runner refuses to resume with different settings and refuses to overwrite a completed run.
