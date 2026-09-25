# SynthAug-Bench

## When Does Diffusion-Based Synthetic Data Augmentation Help Classification, Object Detection, and Image Captioning?

### A Benchmark Across Generator Fidelity, Data Scarcity, Model Architectures, and Vision Tasks

**Original proposal and advisor:** Dr. Amir Jafari  
**Program:** Data Science, The George Washington University, Washington, DC

The classification study covers four data tiers and four CNN/Transformer architectures. The project extends the same question to object detection and image captioning on MS COCO 2017, two task tracks that use the same dataset release, and to a study of pretrained features and hybrid modeling.

This document describes the research plan. The Python environment and a GPU calculation with gradients have been checked on the A10G. CIFAR-100 has been downloaded and checked for image dimensions, class counts, and split sizes. The classification baselines have not been trained yet.

## 1. Objective

The goal of this project is to build a reproducible benchmark that examines when synthetic data augmentation improves computer vision models. We will compare conventional augmentation, GANs, diffusion models trained from scratch, and pretrained diffusion models with and without adaptation.

The main research question is: under what data, model, and task conditions does diffusion-based synthetic augmentation improve performance over real-only training and conventional augmentation, and when does it provide no clear benefit or reduce performance?

We will study this question across controlled few-shot, constructed long-tail, fine-grained, and naturally imbalanced medical datasets. We will also examine whether CNN and Transformer classifiers respond differently to the same synthetic training data.

The classification benchmark is the foundation of the project. The COCO studies will test whether augmentation also helps models locate objects and describe images. The feature study will examine whether pretrained embeddings provide useful auxiliary information when combined with a task model. Downstream performance claims will be based on held-out real-image evaluation.

### Key objectives

1. Implement the five augmentation families: conventional augmentation, class-conditional StyleGAN2-ADA, a small class-conditional DDPM, pretrained Stable Diffusion with class prompts and textual-inversion tokens, and LoRA-adapted Stable Diffusion with a defined DreamBooth-style objective.
2. Complete the four classification data-scarcity tiers, preserving every named dataset and regime.
3. Fine-tune ImageNet-pretrained ResNet-50, ConvNeXt-Tiny, ViT-B/16, and Swin-T and establish measured baselines before comparing synthetic augmentation, and repeat the Stage 1 comparisons with randomly initialized classifiers. Each model will have real-only, RandAugment, Mixup, and CutMix results under the same recorded evaluation protocol.
4. Measure generated-sample quality with FID and Improved Precision & Recall, then evaluate whether these measurements predict downstream benefit. Generator families will not be assigned a quality ranking in advance.
5. Release a config-driven benchmark with experiment records, reproducible notebooks, and a single-GPU reproduction guide.
6. Evaluate generative augmentation for object detection and image captioning on MS COCO 2017, using valid task-specific annotations, controls, and metrics.
7. Investigate pretrained visual features, auxiliary information, and a defined hybrid approach against matched controls.
8. Compare improvements, failures, and computational costs across the three tasks.

### Research questions

| ID | Research question | Planned evidence |
|---|---|---|
| RQ1 / H1 | As measured generator fidelity improves, does classification benefit increase, plateau, or reverse? | Performance changes across generator conditions within the same dataset, real-data budget, classifier, and synthetic ratio |
| RQ2 / H2 | How do ResNet-50, ConvNeXt-Tiny, ViT-B/16, and Swin-T perform before synthetic augmentation, and do the tested CNNs and Transformers differ in the benefit they receive from it? | Baseline results for all four models, followed by paired changes from each model's own real-only and conventional controls, using the same real splits and generated sets |
| RQ3 / H3 | Do FID and Improved Precision & Recall predict downstream augmentation value? | Quality-to-utility regression within compatible settings, with held-out settings for any predictive claim |
| RQ4 / H4 | Which data-scarcity, generator, and synthetic-ratio settings reduce classification performance relative to conventional augmentation? | Negative and inconclusive changes reported with uncertainty, including rare-class and HAM10000 results |
| RQ5 | Do the augmentation methods tested across classification, detection, and captioning produce consistent benefits across tasks? | Changes from each task's controls, considered alongside annotation validity, training-data quantity, and computational cost |
| RQ6 | Do auxiliary pretrained features and a hybrid model improve performance beyond image augmentation alone? | Matched comparisons with and without auxiliary features, on real-only and real-plus-synthetic training data, including added cost |

RQ1-RQ4 are the original H1-H4 analysis. RQ5 and RQ6 extend the study to detection, captioning, and pretrained features. An improvement, a decline, or an inconclusive result is a possible outcome for each question.

### Execution in stages

The plan is built in five stages. Each stage starts only after the previous one is complete, with results. The phases in Section 4 describe the methods; the stages below set the order in which they are run. Any stage not completed by the December submission is reported as future work.

| Stage | Content | Questions |
|---|---|---|
| 1. Core classification benchmark | CIFAR-100 with 5, 10, 20, and 50 images per class and the full training set; ResNet-50 and ViT-B/16; real-only, RandAugment, pretrained Stable Diffusion with class prompts, and LoRA-adapted Stable Diffusion; synthetic:real ratio 1:1; training seeds 0, 1, 2; generator registry and run matrix. The matrix runs first with ImageNet-pretrained classifiers, then again with randomly initialized classifiers | RQ2, RQ4 on CIFAR-100; effect of classifier initialization |
| 2. Full CIFAR-100 benchmark | ConvNeXt-Tiny and Swin-T; Mixup and CutMix; textual-inversion generation, with a DA-Fusion reproduction on a matching published protocol first; ratios 2:1 and 5:1; SD-Turbo/LCM samplers as separately recorded configurations; seeds 3 and 4; FID and Improved Precision & Recall | RQ2, RQ4 complete on CIFAR-100 |
| 3. Remaining classification tiers | Tier B (CIFAR-100-LT), Tier C (Flowers-102, CUB-200-2011, Oxford-IIIT Pets), Tier D (HAM10000) | RQ4 across tiers |
| 4. Generators trained from scratch | StyleGAN2-ADA and the small DDPM on Tiers A and B; quality-to-utility analysis | RQ1, RQ3 |
| 5. Detection, captioning, and hybrid study | MS COCO 2017 detection and captioning; pretrained features and hybrid modeling | RQ5, RQ6 |

## 2. Dataset

The classification study uses six named datasets across four tiers. Detection and captioning use MS COCO 2017 with different annotation types. Dataset versions, access terms, preprocessing, and split procedures will be recorded before use.

| Tier | Dataset | Planned setting |
|---|---|---|
| A: Controlled few-shot | [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) | 5, 10, 20, and 50 training images per class and the full training set; preserve the official test set |
| B: Controlled long-tail | [CIFAR-100-LT](https://github.com/richardaecn/class-balanced-loss#datasets), constructed from CIFAR-100 | Exponential long-tail splits with imbalance ratios of 100, 50, and 10; include a reweighting comparison |
| C: Fine-grained low-data | [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | Documented low-data training, validation, and test protocol |
| C: Fine-grained low-data | [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) | Documented low-data training, validation, and test protocol |
| C: Fine-grained low-data | [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) | Documented low-data training, validation, and test protocol |
| D: Naturally imbalanced medical data | [HAM10000](https://doi.org/10.7910/DVN/DBW86T) | Skin-lesion classification; keep images of the same lesion in one split |
| Detection extension | [MS COCO 2017: Object Detection](https://cocodataset.org/#download) | Low-data object detection using object categories and bounding boxes; reserve val2017 for final evaluation |
| Captioning extension | [MS COCO 2017: Image Captioning](https://cocodataset.org/#download) | Low-data image captioning using human-written reference captions; reserve val2017 for final evaluation |

### COCO split and annotation plan

COCO provides object-instance annotations and image captions. The detection task covers 80 object categories. Captioning uses the human-written reference captions supplied with the dataset. The two tracks share the image source, but they require separate targets, models, and evaluation procedures. [COCO detection task](https://cocodataset.org/dataset/detection-2017.htm), [COCO captions paper](https://arxiv.org/abs/1504.00325)

We will divide train2017 into fixed, disjoint training and tuning pools. Low-data subsets will be drawn only from the training pool. Both COCO tracks will share the split manifest, and all captions and object annotations for an image will stay in that image's split. Any task-specific exclusions will be recorded. Subset sizes will be fixed after the pipeline pilot and before the main comparisons, with image counts and detection class coverage recorded for each subset.

The official val2017 split will serve as the project's held-out real-image evaluation set. It will not be used to fit generators, learn tokens, choose prompts, set filtering thresholds, select checkpoints, or tune models. This is our study protocol. It is not the COCO hidden test set or the commonly used Karpathy captioning split. Comparisons with published results will require matching the paper's release, image IDs, training data, and evaluation settings. [Captioning split reference](https://cs.stanford.edu/people/karpathy/deepimagesent/)

For COCO, synthetic:real ratios will count unique training images. Caption pairs and object instances will be reported separately. Multiple captions for one image will not be counted as multiple independent training images, and detection image counts will not be described as per-class shot counts.

### Data and generator preparation

We will use fixed training, validation, and test splits, with recorded seeds for few-shot and long-tail sampling. Official test splits will be preserved where provided. Any validation subset created from official training data will be reserved before low-data sampling or generator fitting. We will report the validation-data budget separately from the number of real images used for training. Classification loaders will share a common interface. Detection and captioning loaders will retain their task-specific annotations while using a shared provenance format.

For the CIFAR-100 experiments, the split configuration uses seed 42, reserves 50 validation images per class from the official training pool, and selects 5, 10, 20, or 50 training images per class from the remainder. The full-data budget uses all 45,000 remaining images. We will save the image indices and use nested training subsets, so each larger budget contains the smaller one. The validation set has 5,000 labelled images, separate from the training budget. For example, the 5-shot setting uses 500 training images plus those 5,000 validation images. We will report both counts and will not compare this directly with a protocol that permits only five labelled images per class in total. The official 10,000-image test set stays unchanged.

The long-tail split needs its own recorded class counts and validation design. Reserving validation images changes the available training pool, so the few-shot split will not automatically be reused as an exact reproduction of a published CIFAR-100-LT protocol. We will settle and record that construction before the Tier B runs while retaining all three imbalance ratios and the reweighting comparison.

For each dataset and generator condition, we will record the real and synthetic sample counts, synthetic:real ratio, fitting time, generation time, preprocessing, prompts, checkpoint revision, and quality measurements. Generator adaptation, learned embeddings, and source images or captions for synthetic training data will use training data only. Tuning data may guide method selection, but will not be added to the training pool. Final evaluation data will remain held out until the protocol is fixed.

Generated images are resized to the native resolution of the real images in the same dataset (32×32 for CIFAR-100) before training, and then pass through exactly the same preprocessing as the real images. Pretrained Stable Diffusion produces sharp 512×512 images while CIFAR-100 images are 32×32, so without this step a classifier could separate real from generated images by sharpness rather than content.

Preprocessing will be documented in a notebook for each tier and task. HAM10000 preparation will include lesion-group and duplicate checks; lesion grouping does not establish patient-level independence when patient identifiers are unavailable. The classifiers start from ImageNet-pretrained weights. CUB-200-2011 has documented overlap with ImageNet, so Tier C results on CUB will be checked and reported with that exposure in mind, and the pretraining data of every classifier, generator, task model, and auxiliary encoder will be recorded. Unknown pretraining overlap will remain a limitation. [HAM10000 paper](https://arxiv.org/abs/1803.10417), [CUB-200-2011 notice](https://www.vision.caltech.edu/datasets/cub_200_2011/)

Detection samples will be checked for valid boxes and object labels. Captioning samples will be checked for agreement between the image and its description. A generation prompt is not automatically a correct annotation. Synthetic records will include source training-image or training-caption IDs when those inputs are used, class conditions where applicable, generation settings, and annotation-review records.

## 3. Rationale

Limited training data can make it difficult to learn rare classes, distinguish similar categories, or generalize to a different image domain. Synthetic data offers a way to expand the training set, but more images do not automatically provide more useful information. Generated examples can repeat existing patterns, change class-defining details, or introduce incorrect annotations.

SynthAug-Bench will examine these effects under a shared experimental protocol. The study will connect sample-quality measurements with downstream performance, architectural differences, data scarcity, and computational cost. The related-work review will cover DA-Fusion, ALIA, StableRep, Azizi et al., He et al., and Sariyildiz et al., among others. It will record their exact datasets, splits, model initialization, training budgets, and any published corrections. The benchmark's contribution will be assessed against that review; using these datasets alone does not establish novelty.

HAM10000 provides a required domain-mismatch case study. We will compare pretrained and adapted generation without assuming that adaptation must improve results or that a pretrained model has never encountered similar images.

Detection and captioning extend the question beyond classification. A generated image may preserve its category while changing object position, count, or attributes. COCO lets us study spatial and language supervision using a common image source. The feature and hybrid study will test whether pretrained representations provide additional information under a controlled comparison.

The extended review will include work on [diffusion augmentation for object detection](https://openaccess.thecvf.com/content/WACV2024/html/Fang_Data_Augmentation_for_Object_Detection_via_Controllable_Diffusion_Models_WACV_2024_paper.html) and [diffusion augmentation for image captioning](https://arxiv.org/abs/2305.01855). Their results will be interpreted within their published protocols rather than assumed to transfer to this study.

## 4. Approach

### Phase 1: Foundations and baselines

Prepare the four-tier classification pipeline, the COCO detection and captioning interfaces, and the experiment tracking structure. Create generator_registry.csv and run_matrix.csv before the main experiments. Each planned run will identify its research question, task, dataset release, split, real-data budget, generator, task model, feature condition, synthetic ratio, seed, status, and cost. The planned project structure includes data, generators, classifiers, task models, metrics, benchmarks, and notebooks.

Implement real-only training, RandAugment, Mixup, and CutMix for classification. Evaluate the three named conventional policies with separately recorded settings and results. Train all four classification architectures and include the long-tail reweighting comparison. Establish separate COCO task controls, the evaluation metrics below, and the five-seed statistical reporting protocol.

#### Four-classifier baseline

We will start the baseline on CIFAR-100 with one CNN and one Transformer, then add the second CNN and Transformer in Stage 2. A working ResNet-50 run alone does not complete this baseline.

| Model | Family | Initialization | Required controls |
|---|---|---|---|
| ResNet-50 | CNN | ImageNet-pretrained, fine-tune all layers | Real-only, RandAugment, Mixup, CutMix |
| ConvNeXt-Tiny | CNN | ImageNet-pretrained, fine-tune all layers | Real-only, RandAugment, Mixup, CutMix |
| ViT-B/16 | Transformer | ImageNet-pretrained, fine-tune all layers | Real-only, RandAugment, Mixup, CutMix |
| Swin-T | Transformer | ImageNet-pretrained, fine-tune all layers | Real-only, RandAugment, Mixup, CutMix |

For this study, the real-only control will use deterministic image preprocessing without stochastic image augmentation. RandAugment, Mixup, and CutMix will each be a separate condition added to that preprocessing. Their settings and application probabilities will be recorded. Mixing these policies will require a separate named condition rather than changing a baseline silently. Generated images are added to the real-only condition's training set, so a synthetic-data condition differs from real-only only by the added images.

The first development pass will use 50 training images per class to check the loader, classifier head, gradients, validation scores, saved checkpoint, and measured runtime for each model. These checks will use training and validation data. The complete CIFAR-100 baseline then covers the four shot counts, the full training set, and training seeds 0, 1, 2, 3, and 4. Four models, five data budgets, four controls, and five seeds give 400 planned downstream training runs, excluding development runs and tuning. Real-only runs account for 100 of these. The random-initialization arm repeats the Stage 1 part of this matrix. We will measure runtime before scheduling the full set on the single GPU.

We will use the same shared training and evaluation code for all four models, with the model and augmentation policy selected through configuration. The baseline does not need four separate training scripts or a custom model framework. We will extend that pipeline to the remaining classification datasets, retaining the required long-tail, fine-grained, and medical comparisons. Finishing CIFAR-100 completes one baseline stage, not all four data tiers.

All four classifiers start from ImageNet-pretrained torchvision weights, with the output layer replaced by a head for the dataset's class count, and every layer is fine-tuned. The exact weight versions are recorded (for example `IMAGENET1K_V2` for ResNet-50 and `IMAGENET1K_V1` for ViT-B/16). Images are resized to 224×224 and normalized with the ImageNet mean and standard deviation. For a given model and training seed, its real-only, conventional, and synthetic-data runs start from the same pretrained weights and the same seeded head initialization. Each condition will train independently; a synthetic-data run will not continue training from its completed real-only control. We will record the model implementation version, weight version, initialization seed, input resolution, interpolation, normalization, optimizer, and schedule. [Torchvision model weights](https://docs.pytorch.org/vision/stable/models.html)

The fine-tuning recipe is AdamW with weight decay 0.05 and label smoothing 0.1, a learning rate of about 1e-4 for CNNs and 5e-5 for Transformers confirmed in a pilot, linear warmup followed by cosine decay, one fixed batch size, and bf16 mixed precision. In the pretrained arm, classifiers, generators, and auxiliary encoders all carry outside pretraining (ImageNet for the classifiers, the Stable Diffusion training data for the pretrained generators). We will record their external training data separately so no result is described as a study with no outside knowledge.

Each Stage 1 comparison is also repeated with randomly initialized classifiers (`weights=None`, every layer trained). Without ImageNet pretraining, the generator is the only source of outside knowledge, so this arm tests whether synthetic data helps more when the classifier starts with none. It uses the same splits, conditions, generated sets, and seeds as the pretrained arm. Training from random weights needs a much longer schedule, so this arm has its own optimizer settings and step budgets, fixed in a pilot and identical across its conditions. The pretrained arm runs first. A short random-initialization pilot (ResNet-50, 50 images per class, one seed) at the start of Stage 1 checks that recipe and measures its runtime.

We will allow model-specific optimizer settings under the same predefined tuning budget. Fairness does not require forcing one learning rate onto all four architectures. The tuning space, selection rule, effective batch size, optimizer steps, and training resolution will be fixed before the main comparisons. We will keep the named architectures intact and document any input resizing; changing ViT-B/16's patch size, for example, would change the model under study. Differences in parameter count and training recipes will be reported when comparing models. Four tested architectures do not establish a universal ranking of CNNs and Transformers.

#### Metrics fixed before comparison

We will define and check the classification metrics before the baseline runs. All four models and all later augmentation conditions will use the same metric implementation, class order, and evaluation split.

| Measure | Definition and use |
|---|---|
| Top-1 accuracy | Fraction of evaluation images assigned the correct class. This is the primary classification measure. |
| Macro-F1 | Calculate F1 for each class, then take the unweighted mean. This shows performance across classes without weighting the average toward frequent classes. |
| Balanced accuracy | Mean recall across the evaluation classes. This helps interpret results when the evaluation set is imbalanced. |
| Per-class recall and class counts | Fraction of each class identified correctly, reported with training and evaluation counts. These results show whether classes with little training data benefit or lose performance. |
| Confusion matrix | Counts of true classes against predicted classes, using a fixed class order. This identifies recurring classification errors. |
| Validation loss | Mean cross-entropy on real validation images with their original labels. We will use it to inspect training and break checkpoint-selection ties, not as a replacement for the primary measure. |
| Runtime and GPU memory | Elapsed training and evaluation time, peak allocated GPU memory, and the hardware and measurement settings. These show the cost of each result. |

We will report accuracy, balanced accuracy, macro-F1, and recall on a 0-100 scale in result tables. Macro-F1 will be the mean of the class-level F1 scores, not F1 computed from averaged precision and recall. We will retain the full expected class list, set undefined F1 contributions to zero, and report any missing evaluation class. Evaluation splits should contain all classes; a missing class requires an explicit limitation rather than an apparently complete score. [Macro-F1 definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), [balanced accuracy definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)

On CIFAR-100's balanced official test set, top-1 accuracy and balanced accuracy are equal. This also holds when training uses a long-tail subset but evaluation still uses the balanced test set. We will not treat the two scores as independent evidence of improvement. Per-class recall and macro-F1 provide additional information about errors and the classes affected.

For each training run, we will select the checkpoint with the highest validation top-1 accuracy. Ties will use lower validation cross-entropy, then the earlier checkpoint. All conditions will use the same evaluation frequency and checkpoint rule. Training loss from Mixup or CutMix uses mixed labels, so we will not rank models by comparing those training losses directly with ordinary real-only training loss.

We will save one result record per run with the model, dataset, split identifier, training budget, initialization, augmentation policy, seed, checkpoint, scores, and measured cost. Saved predictions will include image IDs and true and predicted labels so the classification scores and paired analyses can be checked again. Metric checks will include a small hand-calculated example, perfect predictions, and a class that receives no correct predictions. These are checks to implement, not baseline results.

#### Measuring improvement

Every synthetic-data condition will be compared with that same classifier's real-only result and its conventional controls at the same real-data budget. We will report the separate RandAugment, Mixup, and CutMix results. If one policy is used as the main conventional reference, we will select it from the mean validation accuracy across the seeds for that model and data budget, before examining final test scores. A tie will use the fixed order RandAugment, Mixup, then CutMix.

For accuracy, the paired change is the augmented score minus the control score for the same training seed, expressed in percentage points. We will use the same pairing for other classification scores on their declared scale. We will report the individual paired changes with their mean and sample standard deviation. With five seeds, we add a 95% percentile bootstrap interval using 10,000 resamples of the seed pairs and a recorded analysis seed; that interval measures training-seed variation conditional on the chosen split and generated data. Three seeds allow only ten distinct seed resamples, so Stage 1 reports the three paired values, their mean and standard deviation, and a 95% bootstrap interval over test images for each comparison, with 10,000 resamples and a recorded analysis seed. Neither interval alone supports a universal claim across the full experiment matrix.

The test set will be used only after the training and selection protocol is fixed. If the protocol changes after we inspect test results, the change and the earlier test exposure will be recorded. We will report negative and uncertain changes as well as gains. A higher raw score from one architecture does not show that synthetic augmentation helped it more; the comparison must use the change from its own matched control.

Within a task and data budget, use the same real splits, model initialization policy, optimizer-step budget, and tuning procedure for comparable conditions. Record batch size, real/synthetic sampling probability, image exposures, and any unavoidable differences. Keep a conventional policy unchanged when adding synthetic data to it, so that comparison measures the contribution of the added images. Reuse generated sets across classifiers when the split and generation settings match. A fixed epoch count alone will not be treated as equal training effort when dataset sizes differ.

The planned software includes PyTorch, Hugging Face diffusers/transformers/accelerate/peft, timm, StyleGAN2-ADA, clean-fid, pycocotools, and a compatible COCO caption-evaluation implementation. Tested versions will be pinned during environment setup. The hardware target is one NVIDIA A10G with 24 GB VRAM on AWS g5.2xlarge. Runtime and memory must be measured before claiming that the expanded study fits the available budget.

### Phase 2: GAN baseline

Train class-conditional StyleGAN2-ADA from scratch on the low-data splits. Begin with the Tier A/B experiments at 32x32 or 64x64 resolution. Monitor training stability, sample diversity, and possible collapse.

Generate synthetic sets at ratios of 1:1, 2:1, and 5:1 relative to real training data. Retrain all four classifiers for the applicable settings and record FID, precision/recall, downstream performance, and cost.

### Phase 3: Diffusion trained from scratch

Implement a small class-conditional DDPM with a lightweight U-Net and a noise-prediction objective. Use the same initial Tier A/B scope and resolution as the GAN comparison, with DDIM sampling to reduce generation time.

Generate data at the same three ratios and retrain the four classifiers. Compare GAN and DDPM results under documented, matched from-scratch compute budgets. Measure fidelity rather than assuming that one generator family must produce better samples.

### Phase 4: Pretrained foundation diffusion

Implement class-prompt augmentation with pretrained Stable Diffusion and a separate DA-Fusion-style condition using learned textual-inversion tokens. The diffusion backbone can remain frozen while token embeddings are fitted, so learned-token augmentation will not be labelled entirely training-free.

Reproduce DA-Fusion using a matching published dataset and protocol before extending it to the required classification settings, including fine-grained and medical data. Flowers and Pets remain required project datasets; they will not be called reproduction datasets without a matching published protocol. The authors' COCO-derived classification experiment is distinct from our COCO detection and captioning tracks. [DA-Fusion implementation and token fitting](https://github.com/brandontrabucco/da-fusion)

### Phase 5: Adapted foundation diffusion

Adapt Stable Diffusion across Tiers A-D using LoRA and an explicit DreamBooth-style class or subject-binding objective. Each real-data budget gets its own adapter, fitted only on that budget's training images. Record the trainable components, LoRA rank, adaptation data, and optimization settings. Use the rank range of 4-16 as planned settings to investigate.

Measure memory and runtime with mixed precision and gradient accumulation where supported. Evaluate the proposed SD-Turbo/LCM acceleration options as separately recorded configurations so a checkpoint or adapter change is not hidden inside a sampler comparison.

Generate the required synthetic ratios and complete downstream comparisons across all four classifiers. The HAM10000 analysis will test whether adaptation changes the usefulness of generated data in the medical domain.

### Task and feature extensions

**Object detection on MS COCO 2017.** Establish a detector with a documented backbone, initialization, image resolution, and training budget. Compare real-only training, conventional augmentation that correctly updates boxes, pretrained diffusion augmentation, and adapted diffusion augmentation. Any added generation condition will have its own recorded configuration. The same detector and evaluation protocol will be used within each comparison.

Generated detection images must preserve or receive corrected category and box annotations. Checks will cover moved, added, removed, and partially visible objects. Original boxes will not be reused automatically after an edit. Rejected examples, annotation corrections, automated checks, manual review, and their costs will be recorded. The audit procedure will be fixed on development data before the main runs.

**Image captioning on MS COCO 2017.** Establish a captioning model with documented visual and text components, initialization, decoding settings, and training budget. Compare real-only training, caption-preserving conventional augmentation, pretrained diffusion augmentation, and adapted diffusion augmentation. Generated image-caption pairs will be checked for objects, attributes, counts, and relationships. Crops or flips that change the meaning of a caption will require a corrected label or exclusion.

All reference captions for an evaluation image will remain together. Caption generation and scoring settings will be fixed before final evaluation. Automatic scores will be accompanied by a predefined, blinded sample review of factual agreement with the image. The review will report its sample size and disagreements between reviewers; it will not be presented as an exhaustive check of every generated sentence.

**Pretrained features and hybrid modeling.** The planned comparison will combine a task model's visual features with embeddings from a frozen auxiliary image encoder. We will document the feature dimensions, combination rule, prediction head, trainable components, and inference inputs. Four conditions will separate the effects: the base model trained on real images, the base model trained on real-plus-synthetic images, the hybrid trained on real images, and the hybrid trained on real-plus-synthetic images. The hybrid will also have a capacity-matched control so an enlarged prediction head is not the only explanation for a gain. Evaluation inputs will contain only information available at prediction time.

The detector, captioner, auxiliary encoder, fusion implementation, feature-study coverage, and checkpoint revisions will be selected and recorded after compatibility and pretraining checks. The choice of COCO is settled; these implementation choices are still open. Public checkpoints will not be assumed to be free of COCO or other evaluation-image exposure. Each comparison will state whether it measures low-data adaptation of a pretrained model or learning from a restricted training set.

The COCO comparisons will use fixed low-data subsets, synthetic:real image ratios of 1:1, 2:1, and 5:1, and five downstream training seeds for each selected condition. Prompt-only, learned-token, and LoRA settings will remain separate when used. The run matrix will state the selected task-specific generator coverage explicitly. The study does not require every generator on every task. All classification datasets, methods, architectures, ratios, and repeats remain part of the plan, subject to the generator applicability question recorded in Section 7.

### Phase 6: Cross-condition analysis

Assemble the master benchmarking DataFrame and answer RQ1-RQ6, including all four original hypotheses. Every result will link to its run configuration, source split, annotations, model revision, and control.

#### Evaluation measures

| Task or analysis | Primary measure | Supporting measures and checks |
|---|---|---|
| Classification | Top-1 accuracy | Balanced accuracy, macro-F1, per-class recall, and confusion matrices; validation loss for diagnostics and selection ties; paired changes from all four models' controls as defined in Phase 1 |
| COCO object detection | Bounding-box AP averaged over IoU thresholds 0.50 to 0.95 in steps of 0.05 | AP50, AP75, AP by object size, and per-category analysis |
| COCO image captioning | CIDEr, with the scoring variant and implementation recorded | SPICE, BLEU-4, METEOR, ROUGE-L, and blinded factual-agreement review |
| Generated-image quality | FID and Improved Precision & Recall | Reference and generated counts, preprocessing, feature extractor, and sample-count limitations |
| Feature/hybrid study | The selected task's primary measure | Change from matched controls, added parameters, training and inference time, and memory |
| Computational cost | Measured generation, adaptation, and downstream-training time | Peak VRAM, storage, annotation-review effort, and monetary cost where available |

Detection evaluation will use the COCO bounding-box protocol. Caption evaluation will use a tested implementation with fixed tokenization, reference handling, and score scaling. Exact versions will be saved with the results. [COCO detection evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py), [COCO caption metrics](https://github.com/tylin/coco-caption)

Use five downstream training seeds per condition (three in Stage 1, extended to five in Stage 2) and report means, standard deviations, and paired changes from controls. Report 95% bootstrap confidence intervals with the resampling unit and number of resamples specified. Seed-level uncertainty and uncertainty from evaluation-image sampling will be distinguished. If images are resampled, captions and boxes will stay grouped by image, and HAM10000 observations will stay grouped by lesion. Data-selection and generation seeds will be recorded separately; five training seeds do not imply five independent generated datasets. With only five training repeats, intervals and small effects will be interpreted cautiously.

Report each task's change from its real-only and conventional controls on its own metric scale. Accuracy, AP, and CIDEr will not be pooled into one score. The COCO tracks will support comparisons on a shared image source, while comparisons with the classification tiers will also reflect differences in datasets and models. These results will not be treated as isolating the effect of task alone.

Evaluate quality-to-utility regression within compatible settings, including synthetic ratio and architecture family where applicable. Any claim of prediction on new settings will require held-out evaluation settings that do not share the same generated set with the regression's training data. Per-seed reuse of a generated set will be accounted for rather than treated as independent quality measurements.

FID and Improved Precision & Recall describe generated-image distributions; they do not establish that a box or caption is correct. Conventional Mixup and CutMix controls will not be forced into a generator-quality ranking. Quality comparisons will use matched reference and generated sample counts where feasible, with class composition and exceptions recorded. Development quality measurements will use development references. Any final held-out quality analysis will happen after method selection and will not guide further tuning. Clean-FID standardizes preprocessing but does not remove small-sample uncertainty. [Clean-FID project](https://www.cs.cmu.edu/~clean-fid/)

Report failed and inconclusive experiments alongside improvements, with missing runs identified separately from measured negative results. Both students will independently check H1-H4 and the cross-task and hybrid analyses.

### Phase 7: Guidelines, report, and code release

Produce practical generator-selection guidelines based on data scarcity, task, and measured compute. Prepare the figure set: performance versus measured fidelity, FID/precision-recall versus performance gain, CNN versus Transformer benefit, and the HAM10000 domain comparison. Add COCO detection and captioning results, annotation-audit summaries, and the feature/hybrid findings. Plots will identify the real-data budget and the uncertainty of each comparison. Results figures are saved as SVG or PDF, and diagrams as editable draw.io files with an exported SVG.

Write the 8-10-page paper-format report with motivation, related work, methods, RQ1-RQ6 results, the medical analysis, cross-task findings, practical guidance, and limitations. Segmentation remains future work.

Release the config-driven benchmark with the generator registry, complete run matrix, master results, experiment records, split manifests, annotation-audit records, a full-study execution command, notebooks per phase and tier, COCO task notebooks, a cross-condition analysis notebook, pinned tested dependencies, README, and single-GPU setup and reproduction instructions. Dataset access instructions will respect each source's terms. Prepare the demonstration and final presentation.

The paper-submission milestone remains part of the delivery plan, with destination and timing to be confirmed with the advisor. A completed manuscript and a submission are separate from publication acceptance. The next implementation step is to save and check the classification splits, implement the shared metrics and baseline runner, and measure the Stage 1 classifiers before the synthetic-data comparisons.

## 5. Timeline

The semester has eight remaining working sessions. Calendar dates and resource requirements will be confirmed from pilot measurements. There is no class on Oct 13 (Fall Break) or Nov 24 (Thanksgiving).

| Working session | Work |
|---|---|
| Sep 29 | Stage 1 foundations: CIFAR-100 split manifest, shared training and evaluation code, metrics and tests, generator registry and run matrix; pilot runs to fix learning rates and optimizer-step budgets for both initialization arms |
| Oct 6 | Pretrained-arm real-only and RandAugment baselines for ResNet-50 and ViT-B/16; random-initialization baselines start; literature review draft in the LaTeX report |
| Oct 20 | Preliminary presentation: pretrained-arm baselines, random-initialization baselines completed so far, first generated samples, literature review complete |
| Oct 27 | Pretrained Stable Diffusion pool and pretrained-arm comparisons |
| Nov 3 | LoRA adaptation for each data budget and adapted pools |
| Nov 10 | Adapted-diffusion comparisons for the pretrained arm; synthetic-data runs for the random-initialization arm |
| Nov 17 | Random-initialization arm complete; Stage 1 analysis; Stage 2 as far as time allows; vector figures; paper draft |
| Dec 1 | Final paper, code release; unfinished stages recorded as future work |
| Dec 8 | Final presentation and journal submission |

Small pilot runs will check the pipeline and measure cost. They change execution order only. The full dataset, method, architecture, ratio, and five-seed requirements remain part of the plan and are run in stage order. Any resource shortfall will be recorded and addressed through scheduling and resource planning. A three-day code freeze precedes the final submission.

## 6. Expected Number of Students

The project will be completed by two students.

| Responsibility | Student 1 | Student 2 |
|---|---|---|
| Method ownership | Conventional augmentation, StyleGAN2-ADA, small DDPM, FID/precision-recall, generator registry | Pretrained diffusion, textual inversion, LoRA/DreamBooth-style adaptation, four-classifier benchmarking, master result table |
| Task extensions | COCO detection pipeline, box verification, and AP evaluation | COCO captioning pipeline, image-caption verification, and caption evaluation |
| Shared work | Feature/hybrid study, split checks, statistical cross-checks, guidelines, writing, documentation, and presentation | Feature/hybrid study, split checks, statistical cross-checks, guidelines, writing, documentation, and presentation |

Hold weekly integration meetings so every method uses the same recorded training and evaluation protocol. Both students should be able to reproduce and explain the complete pipeline.

## 7. Possible Issues

| Issue | Planned response |
|---|---|
| Single-GPU memory and total runtime | Measure representative jobs; use supported memory optimizations and restartable runs; include generation, adaptation, evaluation, and storage in the budget |
| From-scratch generation cost | Start with the low-resolution Tier A/B protocol, ADA, and DDIM sampling; retain unresolved tier applicability until clarified |
| GAN collapse or diffusion divergence | Monitor training and sample diversity, save checkpoints, and use predefined stopping criteria; retain failure records |
| Medical domain mismatch | Audit generated lesions and compare pretrained/adapted conditions without assuming a favorable result |
| Full experiment count | Enumerate required coverage before the main study, run it in stage order, and reuse generated sets where scientifically valid |
| Real and generated images differ in resolution | Resize generated images to the dataset's native resolution before the shared preprocessing |
| FID reliability with small samples | Use consistent preprocessing, report reference/generated counts and uncertainty, and avoid treating small-sample scores as definitive |
| Incorrect boxes, captions, or class labels | Preserve or regenerate annotations, audit samples, and record rejected examples and review costs |
| COCO split or reference leakage | Keep train2017 training/tuning pools disjoint; reserve val2017 for final evaluation; keep all annotations for an image together |
| Leakage and pretraining overlap | Split before fitting or generation, track source groups and duplicates, investigate CUB/ImageNet overlap, and document classifier, generator, task-model, and auxiliary-encoder pretraining |
| Unfair comparisons with published COCO results | Match dataset release, image IDs, training-data access, model initialization, and scorer settings before comparing reported scores |
| Caption metrics miss factual errors | Pair fixed automatic metrics with a predefined blinded image-caption review and report the review's limitations |
| Apparent hybrid gains from extra capacity | Compare against the base model and a capacity-matched head control; report added training and inference cost |
| Uncertain or missing results | Report uncertainty and failures explicitly; keep unrun experiments separate from measured negative outcomes |
| Library and checkpoint changes | Pin tested dependency versions and exact model revisions, and verify reproduction after environment changes |

**Generator applicability to clarify:** The objectives describe all five families across all four tiers, while the detailed methods restrict from-scratch GAN/DDPM to Tiers A/B. Both statements remain recorded. We will begin with the shared A/B requirement and resolve C/D applicability with the advisor before declaring the full run matrix final.

**Implementation decisions to finish in Phase 1:** The tuning and optimizer-step budgets, CIFAR-100-LT construction, COCO subset sizes, task models and checkpoint revisions, task-specific generator settings, the auxiliary encoder and fusion rule, feature-study coverage, exact metric implementations, and measured run costs. ImageNet-pretrained initialization with a random-initialization arm in Stage 1, 224-pixel input with ImageNet normalization, the fine-tuning recipe, and the classification measures above are settled. The remaining decisions will be documented before the main study. They do not make any dataset or task in the plan optional.
