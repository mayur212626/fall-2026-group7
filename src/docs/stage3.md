# Stage 3: Flowers-102 and HAM10000

Stage 3 repeats the benchmark on two new domains. Each is at a different distance from what Stable Diffusion draws well:

| | Flowers-102 | HAM10000 |
|---|---|---|
| Images | 8,189 flower photos, 102 classes | 10,015 dermoscopy images of skin lesions, 7 classes |
| Distance from SD | close (SD draws flowers well) | far (SD 1.5 draws dermoscopy poorly) |
| Split | official: train 10/class, val 10/class, test 6,149 | by lesion, stratified, seed 42: 70 / 10 / 20 |
| Budgets (images per class) | 2, 5, 10 | 5, 10, 20, 50 |
| SD pool | 50 per class (5,100 images) | 250 per class (1,750 images) |
| Runs (5 seeds) | 600, ~89 GPU hours | 800, ~206 GPU hours |

The protocol is the CIFAR-100 one, with these differences:

- **Stored resolution.** Images are kept at 224×224: short side to 256 (antialiased bicubic), then a center crop. Generated images go through the same function (`stage3_data.prepare_image`).
- **Split by lesion (HAM10000).** One lesion often has several photos. The split keeps every lesion inside one split, so a test lesion is never seen in training. Small budgets use as many different lesions as possible.
- **Checkpoint metric: balanced accuracy.** This is the mean recall over classes; for Flowers-102 it is the standard mean per-class accuracy. Both test sets are imbalanced (HAM10000 is about two-thirds nevus). Balanced accuracy chooses the checkpoint and the learning rate, and it is the main metric in the tables. Plain accuracy is saved as well.
- **AutoAugment uses its ImageNet policy.** CIFAR-100 used the CIFAR-10 policy.
- **Models and conditions.** ResNet-50 and ViT-B/16, pretrained and from scratch. There are 10 conditions:
  - real-only
  - six conventional augmentations: RandAugment, TrivialAugment, AutoAugment, AugMix, Mixup and CutMix
  - three with SD class prompts: SD 1:1, SD + RandAugment 1:1, and SD + RandAugment 5:1
- **Step budgets.** These are the CIFAR-100 ones (Flowers-102 at 2 images per class uses the 5-image steps). Learning rates come from pilot runs on each dataset.

## Files

| File | What it does |
|---|---|
| `src/component/stage3_data.py` | Download (HAM10000), preprocessing, split manifests |
| `src/component/configs/splits/{flowers102,ham10000}.json` | Split manifests (commit them) |
| `src/component/configs/train_{pretrained,scratch}_{flowers102,ham10000}.json` | Step budgets and learning rates |
| `src/component/stage3_jobs.py` | The run list, split evenly across instances |
| `src/component/stage3_results.py` | Tables: scores, conventional augmentation, SD conditions |
| `src/shellscripts/stage3_setup.sh` | Data, SD pool and sample grid |
| `src/shellscripts/stage3_pilots.sh` | Learning-rate pilots (12 runs) and `pick_lr` |
| `src/shellscripts/run_stage3.sh` | The training runs of one part |

`train.py`, `evaluate_test.py`, `generate_sd.py`, `sample_grid.py` and `pick_lr.py` take `--dataset` (or `--metric`); CIFAR-100 stays the default and its runs are unchanged.

## How to run it

Everything runs from the repository root, in tmux, in the pinned `synthaug-bench` environment.

### 1. First instance: setup and pilots (one dataset)

```bash
git switch main && git pull
conda activate synthaug-bench
bash src/shellscripts/stage3_setup.sh ham10000     # ~15 min data + ~30 min pool (Flowers-102: ~90 min pool)
```

Open `reports/Latex_report/fig/ham10000_sd15_prompt_samples.png` and the full-size previews in `data/synthetic/sd15_prompt_ham10000/preview/`. Note any class whose generated images do not show that class.

```bash
bash src/shellscripts/stage3_pilots.sh ham10000    # ~3 h (Flowers-102: ~1.5 h)
```

`pick_lr` writes the chosen learning rates into the two configs and warns if a rate is at the edge of the range tried. Commit the manifest, both configs, the registry row and the sample grid on a branch:

```bash
git switch -c stage3-ham10000-setup
git add src/component/configs/splits/ham10000.json src/component/configs/train_pretrained_ham10000.json \
  src/component/configs/train_scratch_ham10000.json src/component/configs/generator_registry.csv \
  reports/Latex_report/fig/ham10000_sd15_prompt_samples.*
git commit -m "Stage 3 HAM10000: split manifest, SD pool record, learning rates from the pilots"
git push -u origin HEAD
```

If `stage3_data download` cannot reach Harvard Dataverse, download `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip` and `HAM10000_metadata` from <https://doi.org/10.7910/DVN/DBW86T> into `data/ham10000/raw/`, then run `stage3_setup.sh` again.

### 2. More instances

Every instance must use the same pool file (its SHA-256 is saved in every run). So copy the first instance instead of generating the pool again. In the EC2 console, select the first instance and choose **Actions → Image and templates → Create image** with **No reboot** checked. Then launch the other instances from that image.

### 3. Training runs

On every instance, give its part number and the number of parts. Seeds 0–2 come first:

```bash
bash src/shellscripts/run_stage3.sh ham10000 1 3     # instance 1 of 3
bash src/shellscripts/run_stage3.sh ham10000 2 3     # instance 2 of 3
bash src/shellscripts/run_stage3.sh ham10000 3 3     # instance 3 of 3
```

Then seeds 3–4, with the same part numbers:

```bash
SEEDS="3 4" bash src/shellscripts/run_stage3.sh ham10000 1 3
```

The script first prints the estimated hours of every part:

```bash
python -m src.component.stage3_jobs --dataset ham10000 --seeds 0 1 2 3 4 --parts 3 --summary
```

The estimates come from the CIFAR-100 step times. Check them against the first few runs' `train.log`; the 224×224 augmentations run on the CPU and may be slower. `WORKERS` sets the data-loader processes (default 6 of the 8 vCPUs).

### 4. Check progress (validation only)

```bash
find runs/ham10000-pretrained runs/ham10000-scratch -name result.json | wc -l
python -m src.component.stage3_results --dataset ham10000          # validation tables, any time
```

### 5. Test set, once, after every run of every seed is finished

On every instance, since the checkpoints stay there:

```bash
python -m src.component.evaluate_test --dataset ham10000 --root runs/ham10000-pretrained --confirm-protocol-frozen
python -m src.component.evaluate_test --dataset ham10000 --root runs/ham10000-scratch --confirm-protocol-frozen
tar czf ham10000_part1.tgz --exclude='*.pt' runs/ham10000-pretrained runs/ham10000-scratch
```

Copy the `.tgz` files to one machine. Unpack them all into the same `runs/` folder (the run names never collide), then build the tables:

```bash
for f in ham10000_part*.tgz; do tar xzf "$f"; done
python -m src.component.stage3_results --dataset ham10000
python -m src.component.stage3_results --dataset ham10000 --split test
```

The tables go to `reports/Latex_report/tables/stage3_<dataset>_{scores,conventional,sd}_<init>[_test].tex`, with CSVs beside them.

## Tests

```bash
python -m unittest discover -s src/tests -v
```

The Stage 3 data, run-list and results tests use small synthetic inputs and need neither the datasets nor a GPU.
