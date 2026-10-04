# SynthAug-Bench

**When does diffusion-based synthetic data augmentation help classification, object detection, and image captioning?**

SynthAug-Bench is a reproducible benchmark of when synthetic training images help computer vision models. It compares conventional augmentation, GANs, diffusion models trained from scratch, and pretrained Stable Diffusion with and without adaptation. The comparison spans few-shot, long-tail, fine-grained, and medical classification datasets, four CNN and Transformer classifiers, and object detection and image captioning on MS COCO 2017.

The project is built in stages. Stage 1 covers CIFAR-100 with 5, 10, 20, and 50 images per class and the full training set, on ResNet-50 and ViT-B/16, comparing real-only training, RandAugment, pretrained Stable Diffusion, and LoRA-adapted Stable Diffusion. The full plan and all stages are in [proposal.md](proposal.md).

Data Science Capstone (DATS 6501), The George Washington University, Fall 2026. Advisor: Dr. Amir Jafari.

## Status

Stage 1 is in progress:

- Done: the CIFAR-100 split manifest, the training runner with resume and validation-only checkpoint selection, the metrics and tests, the learning-rate and step-budget pilots for both initializations, the real-only and RandAugment baselines for both initializations (120 runs, ImageNet-pretrained and random; tables and figures in the LaTeX report), and the FID and Improved Precision & Recall tool.
- Next: Stable Diffusion and LoRA-SD generated sets, scored with FID and Improved Precision & Recall, and the synthetic-data comparisons.

The run matrix in `src/component/configs/run_matrix.csv` lists every Stage 1 run and its status. Documentation: [setup](src/docs/setup.md), [data splits](src/docs/data-splits.md), [training and results](src/docs/training.md), [generated-image quality](src/docs/image-quality.md).

## Repository layout

```
src/
  component/     classes and functions (data, models, training, evaluation)
  docs/          setup and usage documentation
  shellscripts/  run scripts
  tests/         unit tests
reports/
  Latex_report/     formal report and literature review
  Progress_Report/  working-session progress reports
research_paper/  journal paper (LaTeX and Word)
presentation/    presentation slides (PDF and PPT)
demo/            demo and project video
cookbooks/       notebooks
```

## Setup

The environment is pinned in [environment.yml](environment.yml) (Python 3.12, PyTorch 2.12 with CUDA 12.6). Instructions are in [src/docs/setup.md](src/docs/setup.md).

```bash
conda env create -f environment.yml
conda activate synthaug-bench
```

Raster images and videos are stored with Git LFS. Run `git lfs install` once before cloning or pulling.

Datasets, generated images, checkpoints and run outputs are kept outside Git (`data/`, `runs/`).
