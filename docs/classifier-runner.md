# Classifier development runs

The shared runner supports ResNet-50, ConvNeXt-Tiny, ViT-B/16, and Swin-T with random initialization, 100 output classes, and all layers trainable. It uses the saved CIFAR-100 split and checks that the labels, class names, and indices still match. It loads only the official training pool, including the validation subset reserved from that pool. It never loads the official test split.

Run the checks from the project folder in the AWS environment:

```bash
python -m unittest discover -s tests -v
python -m scripts.train_classifier --model resnet50 --smoke
```

Use `convnext_tiny`, `vit_b_16`, or `swin_t` for the other models. The augmentation choices are `real_only`, `randaugment`, `mixup`, and `cutmix`.

For the full-data reference, use `--full-data` instead of `--shots`. With the current split configuration, this selects all 45,000 images outside the reserved 5,000-image validation split, with 450 training images per class. The runner derives these indices from the same split seed and checks the saved manifest. It does not replace the manifest or change the few-shot subsets. The official test split remains unused. Full-data and few-shot results have different real-data budgets and must be reported separately.

Check the new option before a longer run:

```bash
python -m unittest discover -s tests -v
python -m scripts.train_classifier --model resnet50 --augmentation real_only --full-data --smoke
```

The full-data smoke check selects training batches from the 45,000-image pool but still performs only two updates and evaluates 200 validation images. All 5,000 reserved validation images stay excluded from training. It writes to a separate `resnet50_full_real_only_seed0_smoke` directory. The default development configuration still has 1,000 steps; the full-data flag changes the data budget, not the training duration. A longer full-data schedule must be chosen separately.

Smoke mode performs two optimizer steps on the chosen training subset and evaluates two reserved validation images per class, for 200 validation images. It saves a checkpoint and verifies that reloading it reproduces the saved predictions and validation loss. Smoke scores are implementation checks, not baseline results. Each model and policy gets a separate directory under `runs/`.

The full development mode uses all 5,000 validation images and the step count in `configs/cifar100_training.json`. It selects a checkpoint using validation accuracy, then validation loss, keeping the earlier checkpoint on a tie. Both modes save the configuration, model initialization hash, split hash, validation history, best checkpoint, prediction image IDs and labels, metrics, elapsed time, and peak allocated GPU memory. The source image IDs in the prediction file refer to the original training pool.

The configuration also records the selected training-index hash and optimizer steps per epoch. Each validation entry now includes the mean training cross-entropy over the preceding interval, weighted by batch size. This loss is measured during updates, with the active training augmentation and train-mode layers. It is not a separate evaluation of the saved checkpoint. Mixup and CutMix use soft targets, so their training losses should not be treated as directly comparable to hard-label losses.

The starting settings are 224 by 224 bilinear resizing, constant normalization with mean and standard deviation 0.5 for each channel, float32 tensors, batch size 8, and two data-loader workers. Resizing keeps the named model architectures and ViT-B/16 patch size intact. Normalization uses declared constants rather than statistics from validation or test images. TF32 is disabled. Exact numerical repetition across hardware and library versions is not guaranteed; software versions and relevant settings are saved.

The real-only condition uses deterministic preprocessing. RandAugment is applied to resized images with two operations, magnitude 9, bilinear interpolation, and fill value 128. Mixup uses alpha 0.2 and CutMix uses alpha 1.0; each is applied to every training batch in its own condition. Their soft labels are passed directly to cross-entropy. Training batches are shuffled, and an incomplete final batch is dropped. Validation keeps every selected image and its original label. No stochastic validation augmentation is used.

SGD with momentum is the initial ResNet-50 optimizer; the other three start with AdamW. Exact learning rates and weight decay are in the configuration. Learning rates warm up and then decay with a cosine schedule, with gradients clipped at norm 1.0. Smoke mode shortens warmup to two steps. These settings are for development, not a tuned or finalized baseline recipe. The tuning budget and final step budgets still need to be fixed before the main comparisons.

Training time includes loading and optimizer updates between validation passes. Validation time covers the periodic validation passes. Training wall time also includes logging and best-checkpoint saves, but excludes preparing and writing the resume checkpoint. These timings exclude initial data/model setup and the final checkpoint-reload verification. Resumed runs accumulate completed, checkpointed work; time spent on an interrupted, unfinished epoch is not recoverable from the checkpoint and must be included separately in cost reporting. Peak allocated GPU memory covers the training and periodic validation stage across saved sessions; it is not the total memory reported by `nvidia-smi`.

Existing output directories are refused unless `--resume` is given. For a separate run, provide a new `--output` directory under `runs/`. This runner currently implements the four real-data controls, not synthetic-data loading or the other dataset tiers.

## Training in stages

The new `configs/resnet50_full_development.json` plans 100 epochs at batch size 64, with one validation pass after each epoch and one epoch of learning-rate warmup. The SGD learning rate remains 0.01, with momentum 0.9, weight decay 0.0001, and gradient clipping at 1.0. The 45,000-image pool produces 703 full batches per epoch, or 44,992 image presentations. The incomplete eight-image batch is dropped after shuffling. The total schedule is 70,300 optimizer updates; warmup lasts 703 updates, followed by cosine decay. This is a development recipe, not a tuned or finalized baseline protocol.

The longer schedule starts a new randomly initialized model. The earlier five-epoch pilot saved a best model but no optimizer or random state, so it cannot provide an exact training resume. New runs save `last.pt` after each completed epoch. It contains the latest model, optimizer, CPU and CUDA random states, step, best model and predictions, validation history, and accumulated measurements. The file is written to a temporary path before replacing the previous checkpoint. An interrupted write leaves the previous completed checkpoint available.

Resume support requires whole training epochs and validation exactly once per epoch. Worker processes restart each epoch. The training-loader seed is the training seed plus the zero-based epoch number, which reproduces the epoch's shuffle and worker seeds when starting again. The saved step determines the learning rate within the original schedule. Configuration, source hashes, split, and recorded environment must match before resuming. These measures preserve training state; they do not guarantee identical numerical results across nondeterministic GPU operations or different software and hardware.

After the tests pass, start with one epoch to check the complete save-and-continue path:

```bash
python -m scripts.train_classifier --model resnet50 --augmentation real_only --full-data --seed 0 --config configs/resnet50_full_development.json --output runs/resnet50_full_real_only_seed0_100epochs --stop-after-epochs 1
```

Then continue the same run through epoch two:

```bash
python -m scripts.train_classifier --model resnet50 --augmentation real_only --full-data --seed 0 --config configs/resnet50_full_development.json --output runs/resnet50_full_real_only_seed0_100epochs --resume --stop-after-epochs 2
```

`--stop-after-epochs` is an absolute stopping point within the planned run, not a number of additional epochs. It leaves the original 100-epoch learning-rate schedule intact. Removing that option while retaining `--resume` continues to the planned end. An interruption restarts from the latest completed epoch, with up to one unfinished epoch repeated. The best model, predictions, and history are restored together from that checkpoint, so an interrupted save cannot leave them describing different training steps. `result.json` records whether the planned run is paused or complete.

The resume tests compare continued training against uninterrupted updates for SGD and AdamW with dropout, check restored random states, reject changed settings, and simulate a failed checkpoint write. They are implementation checks, not classifier performance results. The checkpoint format follows the [PyTorch saving and loading guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

The training scripts use the installed Torchvision models and augmentation operations. No new training package is required. [Torchvision models](https://docs.pytorch.org/vision/main/models), [Mixup and CutMix usage](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html).
