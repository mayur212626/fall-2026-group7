# Progress Report

SynthAug-Bench, Data Science Capstone (DATS 6501), Fall 2026. Advisor: Dr. Amir Jafari.
Team: Mayur Patil (MP), Manoj Kodihalli Venkatesh (MKV).

This report is updated every week. Each week lists what was planned, what was done (with the merged pull requests and commits), what comes next, and who did it.

| Week | Focus | Main pull requests |
|---|---|---|
| 1-2 | Repository, proposal and research questions | [#4](https://github.com/mayur212626/fall-2026-group7/pull/4) |
| 3-4 | Repository layout, staged proposal, split manifest, metrics, training runner and tests | [#6](https://github.com/mayur212626/fall-2026-group7/pull/6) |
| 5 | Pilots, 60 pretrained baselines, pipeline diagram, report skeleton, first reviewed PRs | [#6](https://github.com/mayur212626/fall-2026-group7/pull/6), [#7](https://github.com/mayur212626/fall-2026-group7/pull/7), [#17](https://github.com/mayur212626/fall-2026-group7/pull/17) |
| 6 | Synthetic pools, generated-image quality, random-initialization baselines | [#17](https://github.com/mayur212626/fall-2026-group7/pull/17), [#18](https://github.com/mayur212626/fall-2026-group7/pull/18), [#20](https://github.com/mayur212626/fall-2026-group7/pull/20), [#22](https://github.com/mayur212626/fall-2026-group7/pull/22) |
| 7 | Synthetic-data results and literature review | [#24](https://github.com/mayur212626/fall-2026-group7/pull/24)–[#28](https://github.com/mayur212626/fall-2026-group7/pull/28) |
| 8 | Error analysis and uncertainty (Fall break) | [#30](https://github.com/mayur212626/fall-2026-group7/pull/30) |
| 9 | Preliminary presentation | [#32](https://github.com/mayur212626/fall-2026-group7/pull/32), [#33](https://github.com/mayur212626/fall-2026-group7/pull/33) |

---

## Weeks 1-2: Repository, proposal and research questions

**Planned**

- Choose the capstone project and confirm the team with the advisor.
- Create the shared repository.
- Write the project proposal and the research questions.

**Done**

- Chose SynthAug-Bench, proposed by Dr. Jafari, as a two-person project; the advisor approved the project and the team. (MP, MKV)
- Created the repository with a project README: [`174ff79`](https://github.com/mayur212626/fall-2026-group7/commit/174ff79), and added the CIFAR-100-LT dataset link: [`c6db9c9`](https://github.com/mayur212626/fall-2026-group7/commit/c6db9c9). (MP)
- The advisor asked for a wider scope than classification alone. The plan was extended to object detection and image captioning on MS COCO 2017: [`8e9ad9a`](https://github.com/mayur212626/fall-2026-group7/commit/8e9ad9a), [`e771a05`](https://github.com/mayur212626/fall-2026-group7/commit/e771a05). (MP, MKV)
- Wrote six research questions: generator fidelity versus benefit (RQ1), CNNs versus Transformers (RQ2), whether FID and Improved Precision & Recall predict the benefit (RQ3), where synthetic data hurts (RQ4), consistency across tasks (RQ5), and pretrained features and a hybrid model (RQ6). They are in [proposal.md](../../proposal.md). (MP, MKV)
- Trained a first set of four CIFAR-100 classifiers for six epochs over five seeds to check the GPU setup: PR [#4](https://github.com/mayur212626/fall-2026-group7/pull/4) ([`642e8bd`](https://github.com/mayur212626/fall-2026-group7/commit/642e8bd)). These runs were exploratory and were later replaced by the Stage 1 protocol. (MP)

**Next**

- Move the repository to the course capstone template.
- Split the proposal into stages that can be finished one at a time.

---

## Weeks 3-4: Repository layout, staged proposal, split manifest, metrics, training runner and tests

**Planned**

- Set up the GPU environment on AWS and the CIFAR-100 data loading.
- Restructure the repository to the capstone template.
- Rewrite the proposal as a staged plan.
- Build the shared code every run depends on: data splits, metrics, training runner, and tests.

**Done**

- Set up the GPU environment (g5.2xlarge, A10G) and CIFAR-100 loading: [`e6289f1`](https://github.com/mayur212626/fall-2026-group7/commit/e6289f1); fixed the baseline plan and evaluation metrics: [`283571f`](https://github.com/mayur212626/fall-2026-group7/commit/283571f). (MP)
- Reset the repository to the capstone template layout (`src/component`, `src/docs`, `src/shellscripts`, `src/tests`, `reports/`, `presentation/`, `research_paper/`, `demo/`, `cookbooks/`): [`48e53a4`](https://github.com/mayur212626/fall-2026-group7/commit/48e53a4), [`7c5e3fc`](https://github.com/mayur212626/fall-2026-group7/commit/7c5e3fc). (MKV)
- Rewrote the proposal as five stages, each started only when the previous one has results, and rewrote the README as a project overview: [`00194cc`](https://github.com/mayur212626/fall-2026-group7/commit/00194cc). (MP)
- Built the CIFAR-100 split manifest: 50 validation images per class (5,000) taken from the training set, nested budgets of 5, 10, 20 and 50 images per class plus the full 45,000, split seed 42, and a fingerprint of the labels: [`20dd45f`](https://github.com/mayur212626/fall-2026-group7/commit/20dd45f), [`8041957`](https://github.com/mayur212626/fall-2026-group7/commit/8041957), [`819ad7f`](https://github.com/mayur212626/fall-2026-group7/commit/819ad7f). (MP)
- Added the classification metrics (accuracy, macro-F1, balanced accuracy, per-class recall) with hand-checked tests: [`deb987f`](https://github.com/mayur212626/fall-2026-group7/commit/deb987f). (MP)
- Added the training runner with the data pipeline, models, configs, resume, validation-only checkpoint selection, and tests: [`694d129`](https://github.com/mayur212626/fall-2026-group7/commit/694d129). (MP)

All of this was merged in PR [#6](https://github.com/mayur212626/fall-2026-group7/pull/6).

**Next**

- Pilot runs to choose the learning rate and the step budget for each model and initialization.
- Run the Stage 1 pretrained baselines.

---

## Week 5: Pilots, pretrained baselines, pipeline diagram, report skeleton, first reviewed PRs

**Planned**

- Learning-rate and step-budget pilots for both initializations.
- The 60 pretrained baseline runs (real-only and RandAugment; ResNet-50 and ViT-B/16; five budgets; seeds 0, 1, 2).
- A diagram of the study pipeline.
- The LaTeX report skeleton and a verified bibliography.
- Start working through reviewed pull requests.

**Done**

- Pretrained pilots: pilot script and learning-rate override [`56f494e`](https://github.com/mayur212626/fall-2026-group7/commit/56f494e); ResNet-50 learning rate set to 3e-4 [`f5139d1`](https://github.com/mayur212626/fall-2026-group7/commit/f5139d1), [`697176d`](https://github.com/mayur212626/fall-2026-group7/commit/697176d); step budgets set from the pilot [`d6fb96f`](https://github.com/mayur212626/fall-2026-group7/commit/d6fb96f). ViT-B/16 uses 5e-5. (MP)
- Random-initialization pilots: [`9f77ecc`](https://github.com/mayur212626/fall-2026-group7/commit/9f77ecc), [`744e7fd`](https://github.com/mayur212626/fall-2026-group7/commit/744e7fd), [`330b096`](https://github.com/mayur212626/fall-2026-group7/commit/330b096), [`e96847d`](https://github.com/mayur212626/fall-2026-group7/commit/e96847d), [`f5660e3`](https://github.com/mayur212626/fall-2026-group7/commit/f5660e3). Chosen: ResNet-50 SGD with learning rate 1.0, ViT-B/16 AdamW with 3e-4, and step budgets of 2,000 to 50,000 steps. (MP)
- Ran the 60 pretrained baselines with [`5d409ae`](https://github.com/mayur212626/fall-2026-group7/commit/5d409ae). RandAugment improved accuracy over real-only at every budget and in every seed (+7.9 points for ResNet-50 and +10.1 for ViT-B/16 at 5 images per class). The table and figure were added in PR [#17](https://github.com/mayur212626/fall-2026-group7/pull/17) ([`1d55fbf`](https://github.com/mayur212626/fall-2026-group7/commit/1d55fbf)). (MP)
- Results analysis with paired per-seed changes and a vector accuracy figure: [`dc33c6e`](https://github.com/mayur212626/fall-2026-group7/commit/dc33c6e); one-time test-set evaluation, locked until the protocol is frozen: [`c3fd891`](https://github.com/mayur212626/fall-2026-group7/commit/c3fd891); restart of runs that stopped before their first checkpoint and seeding of augmentations: [`4bc2e29`](https://github.com/mayur212626/fall-2026-group7/commit/4bc2e29). (MP)
- Study pipeline diagram (draw.io source, SVG and PDF): [`6784eaf`](https://github.com/mayur212626/fall-2026-group7/commit/6784eaf), added to the report with the classification design figure: [`2c73068`](https://github.com/mayur212626/fall-2026-group7/commit/2c73068). (MP)
- LaTeX report skeleton with a verified bibliography: [`8b01f04`](https://github.com/mayur212626/fall-2026-group7/commit/8b01f04). (MP)
- First reviewed pull requests: [#6](https://github.com/mayur212626/fall-2026-group7/pull/6) (Stage 1 foundations) and [#7](https://github.com/mayur212626/fall-2026-group7/pull/7) (setup guide fix). (MP, MKV)
- Planned the diffusion part of Stage 1 (Stable Diffusion class prompts and LoRA adaptation per budget), which MKV took over from here. (MKV)

**Next**

- Random-initialization baselines (60 runs) and the FID and Improved Precision & Recall tool (MP).
- Stable Diffusion class-prompt pool, LoRA adapters for each budget, and the synthetic-data runs (MKV).
- Literature review in the LaTeX report (MKV).

---

## Week 6: Synthetic pools, generated-image quality, random-initialization baselines

**Planned**

- Random-initialization baselines (60 runs) and the FID and Improved Precision & Recall tool.
- Stable Diffusion class-prompt pool, one LoRA adapter per budget, and the synthetic-data runs.

**Done**

- Exported the result tables as CSV and LaTeX and added the pretrained baseline table to the report: PR [#17](https://github.com/mayur212626/fall-2026-group7/pull/17); training progress logged to `train.log`: PR [#18](https://github.com/mayur212626/fall-2026-group7/pull/18). (MP)
- FID and Improved Precision & Recall for generated image sets, and the pilot learning rates recorded: PR [#20](https://github.com/mayur212626/fall-2026-group7/pull/20). (MP)
- Stage 1 run matrix, generator registry and the 60 random-initialization baselines: PR [#22](https://github.com/mayur212626/fall-2026-group7/pull/22). (MP)
- Generated the Stable Diffusion 1.5 class-prompt pool: 45,000 images (450 per class), 8 prompt templates and a negative prompt, DPM-Solver++ with 25 steps and guidance 7.5, reduced to 32×32. (MKV)
- LoRA rank pilot (rank 4, 8, 16; rank 8 chosen on validation accuracy), then one rank-8 adapter per data budget, trained only on that budget's real images, and its pool: 53,500 images. (MKV)
- Scored every pool with FID and Improved Precision & Recall against an equal-size real sample, with a real-vs-real floor. (MKV)
- Ran the SD class-prompt and SD + LoRA conditions for both initializations (120 runs). The code was committed in [`a9bb0da`](https://github.com/mayur212626/fall-2026-group7/commit/a9bb0da). (MKV)

**Next**

- Merge the SD and LoRA conditions, add SD + RandAugment, write the results section, and finish the literature review.

---

## Week 7: Synthetic-data results and literature review

**Planned**

- Merge the synthetic conditions and their results.
- SD + RandAugment as a combined condition.
- Literature review in the LaTeX report.

**Done**

- SD class-prompt and SD + LoRA conditions merged: PR [#24](https://github.com/mayur212626/fall-2026-group7/pull/24) (reviewed by MP). (MKV)
- SD + RandAugment condition and the comparison against RandAugment: PR [#25](https://github.com/mayur212626/fall-2026-group7/pull/25). (MP)
- Pretrained SD + RandAugment results (30 runs): PR [#26](https://github.com/mayur212626/fall-2026-group7/pull/26) ([`097a2e3`](https://github.com/mayur212626/fall-2026-group7/commit/097a2e3)). (MKV)
- Synthetic-data results section, tables and figures for both initializations: PR [#27](https://github.com/mayur212626/fall-2026-group7/pull/27). (MP)
- Literature review, Related Work section with 21 cited works: PR [#28](https://github.com/mayur212626/fall-2026-group7/pull/28) ([`7eb0fa8`](https://github.com/mayur212626/fall-2026-group7/commit/7eb0fa8)). (MKV)
- Instructor Review #4 received: [#29](https://github.com/mayur212626/fall-2026-group7/issues/29).

**Next**

- Answer Review #4's statistical critique (intervals, verdict rule) and prepare the preliminary presentation.

---

## Week 8: Error analysis and uncertainty (Fall break, no working session)

**Planned**

- Bootstrap intervals and a sign test for the paired comparisons.
- Per-class and per-superclass error analysis.
- Runner for training seeds 3 and 4.

**Done**

- Error analysis from the stored validation predictions: per-class and per-superclass recall changes, confusion changes, 95% bootstrap intervals over validation images, sign test, the extra-seed runner and CI: PR [#30](https://github.com/mayur212626/fall-2026-group7/pull/30) ([`a864513`](https://github.com/mayur212626/fall-2026-group7/commit/a864513), [`61119c0`](https://github.com/mayur212626/fall-2026-group7/commit/61119c0)). (MP; reviewed by MKV)
- Shared the SD class-prompt pool (`pool.npz`, SHA-256 checked against the registry) so seeds 3–4 and the remaining scratch SD + RandAugment runs can use the same images. (MKV)
- Instructor Review #5 received: [#31](https://github.com/mayur212626/fall-2026-group7/issues/31).

**Next**

- Preliminary presentation, seeds 3 and 4 for the main pretrained cells, and an explanation of the sign flip (Review #5, Critique 2).

---

## Week 9: Preliminary presentation

**Planned**

- Preliminary presentation (Oct 20).
- Slides in `presentation/` and the progress report.
- Seeds 3 and 4.

**Done**

- Preliminary presentation of Stage 1: design, protocol, baselines, synthetic-data results, superclass breakdown and next steps. (MP, MKV)
- Slides in `presentation/` and this progress report (Weeks 1–5): PR [#33](https://github.com/mayur212626/fall-2026-group7/pull/33). (MKV; reviewed by MP)
- Seeds 3 and 4 for the main pretrained cells (48 runs; no verdict changed): PR [#32](https://github.com/mayur212626/fall-2026-group7/pull/32). (MP; reviewed by MKV)
- Replied to Review #5 on [#31](https://github.com/mayur212626/fall-2026-group7/issues/31) and took the sign-flip diagnostic. (MKV)
- Instructor Review #6 received: [#34](https://github.com/mayur212626/fall-2026-group7/issues/34).

**Feedback from the presentation (Review #6)**

- "How good was the generation?" and "Is there any metric that shows the generated samples are helpful?"
- That augmentation helps with little data is expected and is not enough of a contribution. The two non-obvious results are that lower FID does not mean more useful images (LoRA has lower FID but helps less), and that the effect flips with initialization.

**Next**

- Reframe the contribution around which generated-data measure predicts usefulness (RQ3) and why the effect flips with initialization.
- Seed-aware intervals (MP) and the sign-flip and RQ3 diagnostics (MKV).
- Plan tickets with owners for Nov 3 to Dec 1.

---

## Week 10: A third generated set, the three-set RQ3, Stage 2 runs and the test set

**Planned**

- RQ3 with more than two generated sets, and the sign-flip diagnostics.
- Stage 2: a third architecture, synthetic:real ratios, Mixup/CutMix.
- Freeze the protocol and score the test set once.

**Done**

- Third generated set, SD img2img (strength 0.5, edits of each budget's own images), with its pool quality against a reference without the budget's images: issue [#44](https://github.com/mayur212626/fall-2026-group7/issues/44), PR [#49](https://github.com/mayur212626/fall-2026-group7/pull/49). (MKV; reviewed by MP)
- RQ3 over three generated sets with every pair compared: prompts > LoRA > img2img in 46 of 52 comparisons; FID and recall rank the sets backwards; precision is at chance where its verdict varies. Claim scoped to three sets: issue [#42](https://github.com/mayur212626/fall-2026-group7/issues/42), PR [#50](https://github.com/mayur212626/fall-2026-group7/pull/50), follow-ups [#52](https://github.com/mayur212626/fall-2026-group7/pull/52). (MKV; reviewed by MP)
- SD-only condition (generated images without the real ones): PR [#45](https://github.com/mayur212626/fall-2026-group7/pull/45). (MP; reviewed by MKV)
- Stage 2 runs, 186: ConvNeXt-T (learning rates from validation-only pilots), SD at 2:1 and 5:1, Mixup and CutMix (code from [#43](https://github.com/mayur212626/fall-2026-group7/pull/43)). (MKV)
- Protocol frozen, then the test set scored once from each run's best validation checkpoint, Stage 1 and Stage 2. (MKV, MP)
- Instructor Review #7 received: [#51](https://github.com/mayur212626/fall-2026-group7/issues/51); replied by MP.

**Next**

- Stage 2 PR with the test tables; seeds 3 and 4 for Stage 2.
- The 11-03 session asks (see the session notes below).

---

## Week 11: Stage 2 merged, ensembles, stacked augmentation and a GAN

**Planned**

- Merge Stage 2 with five seeds and the test tables.
- The 11-03 asks: a real + synthetic ensemble, more augmentation types, a GAN generator.

**Done**

- Stage 2 merged, all 150 comparisons with five paired seeds on validation and test ([`4cab400`](https://github.com/mayur212626/fall-2026-group7/commit/4cab400), [`34bd369`](https://github.com/mayur212626/fall-2026-group7/commit/34bd369), [`2e20c7f`](https://github.com/mayur212626/fall-2026-group7/commit/2e20c7f)): PR [#53](https://github.com/mayur212626/fall-2026-group7/pull/53), closing [#47](https://github.com/mayur212626/fall-2026-group7/issues/47). The sign flip holds for ResNet-50, ViT-B/16 and ConvNeXt-T; at 5:1, SD beats RandAugment from scratch in 3 of 4 cells. (MKV; reviewed by MP)
- Seeds 3 and 4: 124 Stage 2 runs (MKV) paired with 48 Stage 1 controls, PR [#54](https://github.com/mayur212626/fall-2026-group7/pull/54) (MP; reviewed by MKV, who re-ran one control seed: identical, 71.18 at step 1500). Both sides in the pinned environment.
- Stacked augmentation: TrivialAugment, AutoAugment and AugMix, SD on top of each, SD 5:1 + RandAugment (360 runs). On test at 5 images per class, the best SD method beats the best of six conventional augmentations: 43.0% → 58.0% (ResNet-50) and 57.6% → 70.7% (ViT-B/16) pretrained, 14.4% → 23.9% (ResNet-50) from scratch. PR [#57](https://github.com/mayur212626/fall-2026-group7/pull/57), tickets [#59](https://github.com/mayur212626/fall-2026-group7/issues/59). (MKV; reviewed by MP)
- Ensembles: real + SD against a real + real control; real + SD wins in 23 of 25 settings on test, +6.8 to +8.1 points at 5 images per class pretrained. PR [#57](https://github.com/mayur212626/fall-2026-group7/pull/57), ticket [#58](https://github.com/mayur212626/fall-2026-group7/issues/58). (MKV, with MP's exported predictions)
- Class-conditional StyleGAN2-ADA on the 50-per-class training images (FID 25.9 against them, no copies found): GAN images lower accuracy for pretrained models and raise it from scratch. PR [#56](https://github.com/mayur212626/fall-2026-group7/pull/56). (MP; reviewed by MKV)
- Instructor Review #8 received: [#55](https://github.com/mayur212626/fall-2026-group7/issues/55); replied by MKV.

**Next**

- Merge [#54](https://github.com/mayur212626/fall-2026-group7/pull/54), [#56](https://github.com/mayur212626/fall-2026-group7/pull/56) and [#57](https://github.com/mayur212626/fall-2026-group7/pull/57); GAN decision and paper tickets.
- Paper skeleton in `research_paper/Latex/`, journal shortlist, then results into the report and the paper by Nov 22.

---

## Working-session notes

**Oct 27**

- Deleted the stale `manoj` branch; all its commits were already on `main`. (MKV)
- Opened [#36](https://github.com/mayur212626/fall-2026-group7/issues/36), the sign-flip and RQ3 diagnostics, and its draft PR [#37](https://github.com/mayur212626/fall-2026-group7/pull/37): per-class Improved Precision/Recall, diversity ratio and centroid gap of the SD pool against the per-class recall change. First result: no per-class measure predicts where SD helps, and the one significant SD − RandAugment correlation (scratch ResNet-50) comes from RandAugment's own per-class gain. The real-only classifiers' accuracy on the generated images is pending (it needs the real-only checkpoints). (MKV)
- Seed-aware intervals: a 95% t-interval over training seeds beside the image bootstrap, PR [#35](https://github.com/mayur212626/fall-2026-group7/pull/35) (MP). Reviewed by MKV: re-ran the 26 tests and recomputed two intervals by hand; both matched.

**Nov 3**

- Instructor's notes: adding ConvNeXt-T shows the result is stable but does not improve accuracy. Asks: (1) a GAN generator, (2) a real-data + synthetic-data ensemble, (3) a lot of augmentation; two more seeds on two more instances in parallel; two weeks to improve results, then write the paper with placeholder results.
- Followed up in Week 11: [#56](https://github.com/mayur212626/fall-2026-group7/pull/56) (GAN), [#57](https://github.com/mayur212626/fall-2026-group7/pull/57) (ensembles and stacked augmentation), seeds 3 and 4 ([#54](https://github.com/mayur212626/fall-2026-group7/pull/54), [`34bd369`](https://github.com/mayur212626/fall-2026-group7/commit/34bd369)).
