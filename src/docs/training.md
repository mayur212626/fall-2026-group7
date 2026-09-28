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
2. More learning rates per model at 50 images per class: 3e-5, 3e-4 and 1e-3 for ResNet-50, 2e-5 and 1e-4 for ViT-B/16.
3. ResNet-50 step budgets again at its chosen learning rate.

Validation accuracy at 50 images per class chose the learning rates in `train_pretrained.json`:

| Model | Learning rates tried | Chosen |
|---|---|---|
| ResNet-50 | 3e-5: 61.76, 1e-4: 68.90, **3e-4: 71.44**, 1e-3: 68.38 | 3e-4 |
| ViT-B/16 | 2e-5: 77.02, **5e-5: 79.92**, 1e-4: 80.48 | 5e-5; 1e-4 was within single-seed variation |

The step budget of each data size is the pilot's own length. At the chosen learning rates, both models reached their plateau well before the end of every pilot run:

| Images per class | Steps | Plateau step, ResNet-50 | Plateau step, ViT-B/16 | Best validation accuracy, ResNet-50 / ViT-B/16 |
|---|---|---|---|---|
| 5 | 600 | 270 | 90 | 35.60 / 47.68 |
| 10 | 800 | 480 | 120 | 49.54 / 62.54 |
| 20 | 1,000 | 600 | 250 | 59.94 / 73.12 |
| 50 | 1,500 | 750 | 450 | 71.44 / 79.92 |
| full | 6,000 | 3,600 | 3,600 | 83.84 / 88.88 |

These are pilot values from seed 100 on validation data; they are not results of the study.

```bash
bash src/shellscripts/pilot_pretrained.sh
```

Finished runs are skipped and an interrupted run resumes, so the script can be started again. At the end it prints a summary; to print it at any time:

```bash
python -m src.component.summarize_pilot --root runs/pilot-pretrained
```

The plateau step is the first validation within 0.5 percentage points of the best accuracy. The step budget for each data size is set from it and written into the config file before the main runs.

## Pilot runs, random-initialization arm

`src/shellscripts/pilot_scratch.sh` does the same for classifiers trained from random weights, with pilot seed 100 and validation data only. Training from scratch needs far more steps, so the pilot runs in two phases.

1. Learning rates at 50 images per class, 8,000 steps (about 205 epochs): SGD with momentum 0.9 at 0.03, 0.1, 0.3 and 1.0 for ResNet-50, and AdamW at 1e-4, 3e-4 and 1e-3 for ViT-B/16.
2. Step budgets for the other data sizes at the chosen learning rates: 2,000, 3,000, 4,000 and 50,000 steps for 5, 10, 20 per class and full.

| Model | Learning rates tried, validation accuracy at 50 images per class | Chosen |
|---|---|---|
| ViT-B/16 | 1e-4: 19.38, **3e-4: 19.58**, 1e-3: 11.30 (unstable) | 3e-4 |
| ResNet-50 | 0.03: about 21, 0.1: 22.44, 0.3: 28.00, 1.0: pending | pending |

```bash
bash src/shellscripts/pilot_scratch.sh
python -m src.component.summarize_pilot --root runs/pilot-scratch
```

## Stage 1 baselines, pretrained arm

`src/shellscripts/run_pretrained_baselines.sh` trains real-only and RandAugment runs for both models at every data budget with training seeds 0, 1 and 2, using the learning rates and step budgets in `train_pretrained.json`. That is 60 runs, about 12–13 GPU hours on the A10G. Each seed finishes before the next one starts, runs that are already complete are skipped, and an interrupted run resumes.

```bash
bash src/shellscripts/run_pretrained_baselines.sh
```

Runs are written to `runs/stage1-pretrained/`. To see progress and the best validation accuracy of each run so far:

```bash
python -m src.component.summarize_pilot --root runs/stage1-pretrained
```

## Result tables and figures

`src/component/analyze_results.py` aggregates completed runs. It prints the mean and sample standard deviation over seeds for each model, data budget and condition, and the paired change of RandAugment over real-only per seed. A change is a gain or a loss only when every seed agrees in sign; otherwise it is inconclusive. It also saves an accuracy-versus-data-size figure as SVG and PDF.

```bash
python -m src.component.analyze_results --root runs/stage1-pretrained \
    --figure reports/Latex_report/fig/stage1_pretrained_accuracy
```

`--metric` selects `accuracy` (default), `macro_f1` or `balanced_accuracy`. The scores are best validation scores; test-set results are evaluated separately once the protocol is fixed.

## Protocol

- Images are 32×32 CIFAR-100 images, upsampled to 224×224 (bilinear) and normalized with the ImageNet mean and standard deviation.
- `real_only` applies no augmentation. `randaugment` applies RandAugment (2 operations, magnitude 9) to the 32×32 image before upsampling.
- Training runs for a fixed number of optimizer steps: linear warmup over 5% of the steps, then cosine decay; gradient norm clipped at 1.0; label smoothing 0.1; bf16 mixed precision on the GPU.
- The validation set is scored 20 times at equal step intervals. The best checkpoint has the highest validation accuracy; ties go to the lower validation loss, then the earlier step.
- Every run is seeded (Python, NumPy, PyTorch, deterministic cuDNN), and the sample order and the augmentation of each sample depend only on the seed and the step. A resumed run therefore sees the same batches and augmentations as an uninterrupted one, which the unit tests check on the CPU. On the GPU, some kernels (for example the attention backward pass of ViT-B/16) are not guaranteed to be deterministic, so repeated GPU runs with the same seed can differ slightly; the differences between seeds are measured by the three training seeds.
- Augmentation seeds come from the run seed and the sample position through NumPy's SeedSequence. Runs made before this change, including the Stage 1 pretrained baselines, used the seed plus the sample position (`seed * 1000003 + k`), which gives the same stream as long as a run stays below one million samples; their git commit is recorded in `config.json`.
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
