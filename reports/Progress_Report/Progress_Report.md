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

## Working-session notes

**Oct 27**

- Deleted the stale `manoj` branch; all its commits were already on `main`. (MKV)
- Opened [#36](https://github.com/mayur212626/fall-2026-group7/issues/36), the sign-flip and RQ3 diagnostics, and its draft PR [#37](https://github.com/mayur212626/fall-2026-group7/pull/37): per-class Improved Precision/Recall, diversity ratio and centroid gap of the SD pool against the per-class recall change. First result: no per-class measure predicts where SD helps, and the one significant SD − RandAugment correlation (scratch ResNet-50) comes from RandAugment's own per-class gain. The real-only classifiers' accuracy on the generated images is pending (it needs the real-only checkpoints). (MKV)
- Seed-aware intervals: a 95% t-interval over training seeds beside the image bootstrap, PR [#35](https://github.com/mayur212626/fall-2026-group7/pull/35) (MP). Reviewed by MKV: re-ran the 26 tests and recomputed two intervals by hand; both matched.
