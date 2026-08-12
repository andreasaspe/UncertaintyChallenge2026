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

## Step 2 log — augmentation & resolution

### Run A (`runs/aug224`): augmentation alone — FAILED, worse than baseline
convnext_tiny @224 pretrained, same as `baseline_pretrained` except for the
new `RandomResizedCrop`/`ColorJitter`/`RandomGrayscale` augmentation.

| | acc | ece | nll | brier | auroc |
|---|---|---|---|---|---|
| baseline_pretrained | **.5806** | **.0394** | **1.567** | **.532** | **.872** |
| aug224 | .5523 | .0595 | 1.596 | .567 | .858 |

Worse on all five, and `ood` accuracy fell .573 -> .528.

**Diagnosis — severe overfitting, not bad augmentation:**
```
epoch 1 | train_acc=0.576 | val_acc=0.552  val_nll=1.68
epoch 6 | train_acc=0.898 | val_acc=0.524  val_nll=2.41
```
Val accuracy peaks at **epoch 1** and declines while train accuracy runs to
90%. Early stopping then returns an essentially 1-epoch model. Augmented data
is harder, so after one epoch the augmented model has simply learned less —
augmentation only pays off over many epochs, which this setup never reaches.
The binding constraint is the *training dynamics*, not the augmentation.

### Aspect-ratio hypothesis: REFUTED (measured)
Worry was that train (`RandomResizedCrop`, aspect 0.75-1.33) mismatches eval
(squash 796x448 -> square, 1.78x compression). Tested 4 eval transforms on
both checkpoints — **the current squash is the best of them**:

| eval transform | aug224 acc | baseline_pretrained acc |
|---|---|---|
| **squash 224x224 (current)** | **.5523** | **.5806** |
| Resize256 + CenterCrop224 | .4935 | .4673 |
| Resize224 + CenterCrop224 | .5251 | .5098 |
| full frame native 224x398 | .5468 | .5490 |

=> **Do not switch eval to Resize+CenterCrop.** Centre-cropping discards the
frame edges, and camera-trap animals frequently sit at the edge. Keep the
full-frame squash.

### Image sizes (measured)
Originals are ~796x448 (0.36 MP). Resizing to 224 keeps **14%** of the pixels;
320 keeps 29%. So 224 *is* a heavy downsample — real motivation for trying 320,
but only once overfitting is under control.

### Run B (`runs/aug320_small`): ABORTED
convnext_small @320 was launched then killed at epoch 1 — it carried the same
LR/regularisation settings as run A, so it would have reproduced the same
overfitting at ~18 min/epoch. Revisit resolution/backbone only after the
training dynamics are fixed.

### Run C (`runs/reg224`): regularisation fixed it — new best
Same backbone/resolution as `baseline_pretrained`; changed backbone lr
1e-4->3e-5, head lr 1e-3->3e-4, weight decay 1e-4->**0.05**, patience 5->8,
plus the run-A augmentation.

| | acc | ece | nll | brier | auroc |
|---|---|---|---|---|---|
| baseline_pretrained | .5806 | .0394 | 1.567 | .532 | **.872** |
| **reg224** | **.6296** | **.0367** | **1.403** | **.498** | .849 |

**4 of 5 metrics better.** acc +4.9pts; `ood` acc .573->.591, `id` .588->.670.
Only misclassification AUROC regressed (.872->.849) — note `ood` auroc (.883)
is now *higher* than `id` (.815), a reversal worth watching.

Caveat: **run C had no label smoothing** — a patch bug put it in
`fit_temperature` instead of `train()`, so only the LR/weight-decay changes
took effect. So the whole +4.9pts is attributable to LR + weight decay.

Val accuracy still peaks at **epoch 1** (train acc reaches 95% by epoch 9),
so overfitting is reduced but not solved. Runs D (`ls224`, adds label
smoothing 0.1) and E (`lowlr224`, lr 1e-5/1e-4) probe further.

### Gotchas hit (don't repeat)
- `train.py` has **two** `nn.CrossEntropyLoss()` calls. Line ~188 is inside
  `fit_temperature` and must stay **unsmoothed** — calibration is fit against
  true NLL. Only the one in `train()` (~line 291) takes `label_smoothing`.
- Heredoc-piped scripts (`python - <<EOF`) break `DataLoader(num_workers>0)`:
  the forkserver can't re-import `<stdin>`. Write scratch scripts to a real
  file, and guard with `if __name__ == "__main__":`.
- If training finishes but temp-scaling crashes, `model.pt` still holds the
  best weights — refit T separately rather than retraining.

### Run D (`runs/ls224`): label smoothing 0.1 — mildly HARMFUL, dropped
Identical to reg224 except `--label-smoothing 0.1`.

| | acc | ece | nll | brier | auroc |
|---|---|---|---|---|---|
| reg224 | **.6296** | **.0367** | **1.403** | **.498** | .8493 |
| ls224 | .6253 | .0404 | 1.554 | .504 | .8499 |

Worse on 4 of 5, clearly worse on NLL. The tell: fitted **T=0.93 (<1)** —
smoothing made the model *under*-confident, so temperature scaling had to
sharpen rather than soften. Temperature scaling already handles calibration
post-hoc, so smoothing just caps achievable confidence and costs NLL.
It *did* stabilise training (val acc peaked at epoch 3, curve flat across 11
epochs) — it fights overfitting, just not worth the price. **Don't use ls=0.1.**

### Run E (`runs/lowlr224`): lr 1e-5/1e-4 — best accuracy, worst everything else
| | acc | ece | nll | brier | auroc |
|---|---|---|---|---|---|
| lowlr224 | **.6318** | .0428 | 1.759 | .512 | .823 |

Trained longest (early stop epoch 17, best epoch 9) and reached the highest
accuracy, but NLL 1.759 and AUROC .823 are the worst of the good runs.
Confounded: it also carried ls=0.1. Not worth rerunning — the accuracy edge
over reg224 (+0.2pt) is inside val noise (n=918).

### Scoreboard (val, temp-scaled) — Borda rank across all 5 metrics
| run | acc | ece | nll | brier | auroc | Borda |
|---|---|---|---|---|---|---|
| **reg224** | .6296 | **.0367** | **1.403** | **.498** | .849 | **8 (best)** |
| ls224 | .6253 | .0404 | 1.554 | .504 | .850 | 12 |
| baseline_pretrained | .5806 | .0394 | 1.567 | .532 | **.872** | 14 |
| lowlr224 | **.6318** | .0428 | 1.759 | .512 | .823 | 17 |
| aug224 | .5523 | .0595 | 1.596 | .567 | .858 | 24 |

**`reg224` is the model to beat** (lr 3e-5/3e-4, wd 0.05, full augment, no
label smoothing).

**Open concern:** every new run has *worse* misclassification AUROC than the
old baseline (.849/.850/.823 vs .872). The more accurate models are worse at
knowing when they're wrong. Deep ensembles are the standard fix and target
exactly this metric — likely the highest-value remaining move.

### Run F (`runs/basicaug224`): augmentation ablation — augmentation is ~NEUTRAL
reg224's exact config but `--augment basic` (starter-kit resize+flip). This is
the clean isolation that runs A and C never provided.

| | acc | ece | nll | brier | auroc |
|---|---|---|---|---|---|
| reg224 (full aug) | .6296 | **.0367** | **1.403** | .4983 | .8493 |
| basicaug224 (basic aug) | **.6307** | .0471 | 1.538 | **.4928** | **.8640** |

**They tie on Borda (12 each).** Heavy augmentation helps *calibration*
(ece .037 vs .047, nll 1.40 vs 1.54) and hurts *ranking* (auroc .849 vs .864);
accuracy is a wash. So the augmentation is not the win step 2 assumed — it
trades metrics rather than lifting them. Note basicaug224 overfits far harder
(train acc .99, fitted **T=2.19** vs reg224's 1.23) yet still scores well,
because temperature scaling cleans up most of the overconfidence.

### Run G (`runs/reg320`): KILLED mid-run, but the most promising lead
reg224 config at 320px. Killed at epoch 10; **no checkpoint saved** (only
config.json). Before dying it hit **val_acc .6405 at epoch 5 — the highest
val accuracy of any run so far**, and val_nll 1.41 at epoch 1.
=> Higher resolution looks like the real accuracy lever. **Rerun this.**

### Full scoreboard (val, temp-scaled), Borda across all 5 metrics
| run | acc | ece | nll | brier | auroc | Borda |
|---|---|---|---|---|---|---|
| **reg224** | .6296 | **.0367** | **1.403** | .4983 | .8493 | **12** |
| **basicaug224** | .6307 | .0471 | 1.538 | **.4928** | .8640 | **12** |
| baseline_pretrained | .5806 | .0394 | 1.567 | .5324 | **.8724** | 17 |
| ls224 | .6253 | .0404 | 1.554 | .5038 | .8499 | 17 |
| lowlr224 | **.6318** | .0428 | 1.759 | .5121 | .8231 | 21 |
| aug224 | .5523 | .0595 | 1.596 | .5671 | .8577 | 26 |
