# Uncertainty Challenge 2026 — iWildCam

## What this is
A Summer School competition. We train an image classifier on camera-trap
wildlife photos (iWildCam) and are scored on **five metrics**, not just
accuracy: `accuracy`, `ece` (calibration error), `nll`, `brier`,
`misclassification_auroc`. Leaderboard rank = Borda count across metrics,
so a model that's decent on all five beats one that's great on one and bad
on others. Full framing: `../README.md`. Starter-kit usage: `README.md`.

## Working style — I'm new to Claude Code and new to uncertainty ML
- **Explain in plain language before/while you code.** I don't have a
  background in calibration, ECE, temperature scaling, ensembles, etc.
  When you introduce a technique, give me 2-4 sentences on *what it does
  and why it should help our metrics* before diving into code.
- **Comment code more than you normally would.** Default Claude Code style
  is "no comments unless non-obvious" — for this project, lean toward
  explaining *why* a design choice was made (e.g. why MC dropout needs
  `model.train()` at eval time), since I can't infer intent from ML
  conventions I don't know yet.
- Prefer small, explainable steps over big opaque changes. After each
  experiment, tell me in one line what changed and what number to expect
  to move.
- If a technique has a name (focal loss, deep ensemble, MC dropout,
  Platt scaling...), name it explicitly so I can go learn about it later.

## Data & environment
- Data root: `/mnt/c/data/uncertainty_data/challenge_data`
- Run outputs live under `<data root>/runs/<name>/`:
  - `runs/baseline` — convnext_tiny, **not** pretrained (obsolete, kept for reference)
  - `runs/baseline_pretrained` — convnext_tiny, `--pretrained` (current best)
- Conda env is `ss26` — activate it before running anything
  (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate ss26`).
  The `base` env has no torch/numpy/pandas.
- Run everything as `python -m student.<module>` from the parent directory
  (`/home/awias/code/UncertaintyChallenge2026`), matching the README examples.

## Dataset facts (measured, not assumed)
- `K = 57` classes. All 57 appear in train.
- Split sizes: train 18,929 · val 918 · test_public 3,927 · test_private 4,395.
- **Train is 100% `domain == id`.** Val is 454 `id` / 464 `ood`. Those 464
  images are the *only* OOD supervision available anywhere.
- **The shift is covariate, not open-set**: val-`ood` introduces zero
  unseen classes, so "OOD" means *new camera location*, not *new species*.
  Open-set / novel-class detectors are the wrong tool here.
- **Only mildly long-tailed**: per-class train counts run 400 down to 100
  (4:1), i.e. the organizers rebalanced it. No class has <10 images.
  => class reweighting / focal loss is a *weak* lever here, despite the
  README implying otherwise.

## Results log (val, via `python -m student.eval`)
`overall / id / ood`, all with post-hoc temperature scaling unless noted.

| run | acc | ece | nll | brier | mis-auroc |
|---|---|---|---|---|---|
| baseline (not pretrained), T=3.83 | .222 | .056 | 3.13 | .911 | .631 |
| **baseline_pretrained, T=1.72** | **.581** | **.039** | **1.57** | **.532** | **.872** |
| baseline_pretrained, no temp scaling | .581 | .163 | 1.84 | .575 | .866 |

Per-domain for `baseline_pretrained` (temp scaled):
- `id` : acc .588 · ece .054 · nll 1.518 · brier .514 · auroc .894
- `ood`: acc .573 · ece .058 · nll 1.615 · brier .550 · auroc .853

**Key finding:** ImageNet pretraining nearly closed the domain gap. The
id→ood accuracy drop went from 16.6 points (.306→.140) to **1.5 points**
(.588→.573), and id/ood ECE are now near-identical (.054 vs .058). Whatever
OOD-specific handling we add has very little headroom left to exploit —
measure before building.

Temperature scaling is doing real work: it cuts ECE .163→.039 and NLL
1.84→1.57 without touching accuracy (T never changes the argmax).

## Hard constraints — do not violate
- **Never change the metric formulas in `metrics.py`.** The server scores
  with the identical file; changing it locally just makes local numbers
  lie to us.
- **Keep the `Classifier` contract in `model.py`**: `self.backbone`,
  `self.head`, `forward(x) -> logits (N, num_classes)`,
  `embed(x) -> features (N, embed_dim)`. `train.py`/`eval.py`/`predict.py`
  all depend on this shape — breaking it breaks the whole pipeline.
- **Checkpoint dict format stays**: `state_dict`, `num_classes`,
  `temperature`, `backbone`, `hyperparameters`. `eval.py`/`predict.py`
  read these keys directly.
- **Submission format**: one row per `uid` across both `test_public` and
  `test_private`, columns `uid, p_0, ..., p_{K-1}`, rows summing to ~1
  (tolerance 1e-3). `student.predict` already does this — don't hand-roll
  a different CSV writer.

## Strategy notes (from the README's "what to try")
- Baseline already does temperature scaling (fit on val via LBFGS on NLL).
  Cheap next step: try fitting T on Brier instead of NLL and compare.
- Deep ensembles (train N models, average probs) and MC dropout (dropout
  on at eval time, average T forward passes) are the two techniques most
  likely to move `ece`/`nll`/`brier`/`auroc` together without hurting
  accuracy — good candidates before anything fancier.
- ~~iWildCam is long-tailed — class re-weighting / focal loss~~ — measured
  and mostly false for *this* subset (4:1 ratio, see Dataset facts).
  Deprioritized.
- `student.eval` reports `overall`/`id`/`ood` blocks automatically
  (`evaluate_by_domain`), so every experiment shows its domain split for
  free. Watch whether a technique helps `ood` as well as `id`.
- **Accuracy is the load-bearing metric.** NLL, Brier and misclassification
  AUROC all improve as the classifier gets genuinely better, so raising
  accuracy lifts 4 of the 5 Borda columns at once. Calibration tricks only
  reshape confidences — they never move accuracy at all.

## Anti-patterns for this challenge (learned the hard way)
- **Don't train a separate model (autoencoder, from-scratch ViT) to detect
  OOD.** Only 464 OOD images exist. Reconstruction-based OOD scores key on
  low-level statistics (brightness, IR-vs-colour, background texture), which
  here means they detect *"new location"* rather than *"my classifier is
  about to be wrong"* — and the latter is what every scored metric cares
  about. The fine-tuned backbone's own `embed()` features are already the
  right latent space, for free.
- **Don't train a supervised id-vs-ood classifier on val.** Camera-trap
  images from one location share a near-constant background, so such a
  classifier scores ~99% by memorising val's specific locations and then
  transfers poorly to test's *different* held-out locations. Any OOD score
  should be an unsupervised distance to the *train* distribution, with val's
  labels used only to validate it and fit 1-2 parameters.

## OOD gate result (step 1) — measured, `student/ood.py`
Run against `runs/baseline_pretrained/model_temp_scaled.pt`. Scores are
computed from `model.embed()` features vs the train bank — no new model.

- **Domain detection is weak**: AUROC(score ; ood vs id) peaks at **0.638**
  (kNN, k=10); Mahalanobis 0.585. The fine-tuned features barely separate
  held-out locations — consistent with the 1.5-point accuracy gap.
- **OOD scores are worse error-predictors than max-softmax**: 0.809 (kNN
  k=50) vs **0.872** for max-softmax. They are strongly rank-correlated with
  it (-0.69), so mostly redundant, not complementary.
- **Input-dependent temperature `T(x)=exp(a*s+b)`** vs plain scalar T,
  5-fold CV on val + 2000-sample paired bootstrap. `a` was stably positive
  across all folds (+0.069..+0.089), so the signal is real, but tiny:

  | metric | scalar | linear | delta | verdict |
  |---|---|---|---|---|
  | ece | .0366 | .0415 | +.0049 | not significant |
  | nll | 1.5679 | 1.5635 | -.0044 | not significant |
  | brier | .5325 | .5303 | -.0022 | significant win |
  | mis-auroc | .8723 | .8778 | **+.0055** | significant win |

  Accuracy is unchanged by construction (temperature never moves argmax).

**Verdict: real but marginal — 2 of 5 metrics significantly better, none
significantly worse.** Worth wiring into `predict.py` eventually, NOT worth
further investment. Re-run the gate after accuracy improves; the picture may
change. Effort belongs on accuracy, which lifts 4 of 5 Borda columns at once.
