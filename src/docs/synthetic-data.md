# Synthetic data: Stable Diffusion class prompts (`sd_prompt`) and LoRA (`sd_lora`)

Stage 1 compares real-only and RandAugment with two diffusion conditions at a 1:1 synthetic-to-real ratio. Pretrained Stable Diffusion with class prompts (`sd_prompt`) uses the model as released; the LoRA condition (`sd_lora`) first adapts it to each data budget's real images. Both use the same pool format, training option and analysis.

## Generator

| Setting | Value |
|---|---|
| Model | `stable-diffusion-v1-5/stable-diffusion-v1-5`, fp16; the commit is resolved at generation time and recorded |
| Scheduler | DPM-Solver++ (`DPMSolverMultistepScheduler`), 25 steps, guidance 7.5 |
| Size | 512×512, reduced to 32×32 with antialiased bicubic resampling (the real images' resolution) |
| Prompts | 8 templates, e.g. "a photo of a {class}.", "a close-up photo of a {class}."; image `i` uses template `i mod 8` |
| Class names | CIFAR-100 names with underscores replaced; six ambiguous names are rewritten (`ray` → stingray, `mouse` → field mouse, `plain` → grassy plain landscape, `keyboard` → computer keyboard, `tank` → military tank, `seal` → harbor seal; `cattle` → cow) |
| Negative prompt | text, watermark, logo, cartoon, drawing, illustration, painting, blurry, low quality |
| Seeds | image `i` of class `c`: NumPy `SeedSequence([generation seed, c, i])`; generation seed 0 |
| Safety checker | off, because it silently replaces flagged images with black ones |
| Pool size | 450 per class, 45,000 images |

The generator is used as released; it is not trained on CIFAR-100. It carries outside knowledge from its own training data, which is recorded in the generator registry (`sd_prompt`).

## Order of decisions

Every generation setting above was fixed in code before any image was generated, and every pool was finished before any classifier trained on it; none was chosen by looking at classifier results. The settings were checked only on the quick check images and the previews. The one generator setting chosen with classifier feedback is the LoRA rank, through the validation-only pilot planned in the proposal (pilot seed 100, below). `pool.json` records when each pool was finished, and every run's `config.json` when it started.

## Pool

`generate_sd` writes `data/synthetic/sd15_prompt/pool.npz` (images, labels and each image's index within its class), `pool.json` (every setting above, package versions, GPU, generation time and the pool's SHA-256), full-size previews, and fills the `sd_prompt` row of `src/component/configs/generator_registry.csv` (module `generator_registry`). `data/` is excluded from Git; the registry and `pool.json` settings identify the pool.

A run with `k` real images per class takes the generated images with index `0 … k−1` of each class, so the generated images of a smaller budget are always part of those of a larger one, as with the real budgets.

## Commands

From the repository root, in tmux, when no training job uses the GPU:

```bash
bash src/shellscripts/generate_sd_prompt_pool.sh
```

The script first generates 16 images for two classes to check the setup and print the speed, then the full pool, then the sample figure `reports/Latex_report/fig/sd15_prompt_samples.{svg,pdf,png}`. Each class is saved when it finishes, so the script can be restarted and continues with the next class.

Then the 30 pretrained-arm runs (ResNet-50 and ViT-B/16, five budgets, seeds 0–2):

```bash
bash src/shellscripts/run_sd_prompt.sh
python -m src.component.analyze_results --root runs/stage1-pretrained
```

## Training with generated images

`train.py --condition sd_prompt --synthetic-pool data/synthetic/sd15_prompt/pool.npz` appends `ratio × k` generated images per class (`--synthetic-ratio`, default 1) after the real training images and samples the combined set uniformly, so about half of every batch is generated at 1:1. Learning rate and step budget are those of the real-only run, so the two conditions train for the same number of optimizer steps and pair by seed. `sd_prompt` and `sd_lora` apply no RandAugment. `sd_prompt_randaugment` uses the class-prompt pool and applies RandAugment to every image, real and generated, so its paired change over RandAugment shows what the generated images add to conventional augmentation; `run_sd_prompt_randaugment.sh` trains it (pretrained arm by default, `scratch` for the random-initialization arm).

The pool's SHA-256, the ratio and the number of generated images are stored under `settings["synthetic"]`; the runner refuses to resume a run with a different pool. Real-data runs have no `synthetic` key, so their settings are unchanged. Runs are named `pretrained_<model>_sd_prompt_b<budget>_s<seed>`, with `_r<ratio>` added for ratios other than 1.

`analyze_results` reports every condition's paired change over real-only and writes one LaTeX table per condition (`…_sd_prompt.tex`); `evaluate_test` scores synthetic runs like any other run.

## LoRA adaptation (`sd_lora`)

`train_lora` fits one adapter per data budget, only on that budget's real training images (500 for 5 per class up to 45,000 for the full set); validation and test images are never used.

| Setting | Value |
|---|---|
| Base model | the same SD 1.5 revision as the prompt pool; generation refuses an adapter trained on another revision |
| LoRA | rank from the rank pilot below (default 8, alpha = rank) on the U-Net attention projections `to_q`, `to_k`, `to_v`, `to_out.0`; VAE, text encoder and U-Net frozen |
| Objective | DreamBooth-style class binding: each image is captioned with a pool template and the identifier token before the class name ("a photo of a cfr bear."); standard noise-prediction MSE |
| Input | 32×32 real images upsampled to 512×512 (bicubic), random horizontal flip |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.01, 5% linear warmup, gradient clipping 1.0, batch 8, bf16 autocast |
| Steps | 1,000 / 1,500 / 2,000 / 3,000 / 6,000 for 5 / 10 / 20 / 50 per class / full |
| Seed | 0; batch order, captions, flips, noise and timesteps depend only on seed and step |

Each adapter directory holds `pytorch_lora_weights.safetensors`, `lora.json` (settings, budget, image fingerprint, versions, training time, final loss) and `loss.jsonl`.

`generate_sd --lora-dir data/lora/sd15_lora_b<budget>` loads the adapter and writes prompts with the identifier ("a photo of a cfr bear."), with a negative prompt of only "text, watermark, logo" so the output can match the soft look of CIFAR photos. Each budget's pool has exactly as many images per class as the budget has real images (1:1): 5, 10, 20, 50 and 450. Pools are stored in `data/synthetic/sd15_lora_b<budget>/` and registered as `lora_sd_b<budget>`.

## Image quality: FID and Improved Precision & Recall

`pool_quality` scores a `pool.npz` with the measures and features of `image_quality` (Clean-FID "clean" Inception features of the 32×32 images, FID, Improved Precision & Recall with k = 3), at the size a run uses: the first `k` images per class against a class-balanced real sample of the training pool (never the test set) with the same number of images per class, so FID is always compared at equal sample counts. FID is biased upwards for small sets, so each result also reports its bootstrap SD (10 resamples of the generated set) and, when the pool allows, a real-versus-real floor: the FID of two disjoint real samples of that size.

```bash
bash src/shellscripts/measure_quality.sh
```

The script measures every finished pool (the class-prompt pool at each budget's size and each LoRA pool), writes `reports/Latex_report/tables/generator_quality.csv`, and copies the scores of the full class-prompt pool and of each LoRA pool into the generator registry.

## LoRA rank pilot

The proposal plans ranks 4–16. `pilot_lora_rank.sh` fits rank 4, 8 and 16 adapters on the 50-images-per-class budget, generates a 1:1 pool from each and trains ResNet-50 and ViT-B/16 on real + generated images with pilot seed 100, like the classifier pilots, scoring only validation data. `choose_lora_rank` keeps rank 8 unless another rank has a higher mean validation accuracy over the two classifiers by more than 0.5 percentage points (one pilot seed cannot resolve less), and writes `data/lora/chosen_rank.json`; `train_lora_adapters.sh` fits every budget's adapter at that rank.

## Both classifier arms

`run_sd_prompt.sh` and `run_sd_lora.sh` take the arm as an argument: `pretrained` (default, runs first) or `scratch` (random initialization, with the learning rates and step budgets of `train_scratch.json`). Both arms use the same pools, so the generated data is identical and only the classifier's starting weights differ. The scratch arm pairs with the scratch baselines in `runs/stage1-scratch`.

## Run matrix

`python -m src.component.run_matrix --keep-missing` updates `src/component/configs/run_matrix.csv` (see [training](training.md)) with the status of the runs on this machine; the baseline runs from the other machine keep their status.

## Whole Stage 1 diffusion part in one command

```bash
nohup bash src/shellscripts/run_stage1_diffusion.sh > runs/stage1_diffusion.log 2>&1 &
```

In the proposal's order: quick checks of both generators (16 images of two classes each; a 300-step adapter), the prompt pool and the 30 pretrained-arm `sd_prompt` runs, the LoRA rank pilot, the adapters, the LoRA pools and the 30 pretrained-arm `sd_lora` runs, tables, figure and run matrix (about 35 GPU hours on the A10G); then the 60 random-initialization runs (about 3 GPU days) and their tables. Every step skips finished work, so the same command continues after an interruption.

The paired comparison needs each arm's real-only and RandAugment baselines in the same `runs/stage1-<init>` folder. Test scores follow the usual one-time rule, after the protocol is fixed: `python -m src.component.evaluate_test --root runs/stage1-pretrained --confirm-protocol-frozen`, then `analyze_results --split test`.
