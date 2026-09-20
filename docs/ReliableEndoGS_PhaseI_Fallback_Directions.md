# ReliableEndo-GS — Phase I Fallback Directions

## Purpose

This document defines **two replacement research directions** for ReliableEndo-GS if the current **Phase I on the full SCARED-C development corpus still fails**.

The purpose is not to continue searching for a better heuristic uncertainty score.

The fallback principle is:

> **Stop using heuristic reliability proxies and model predictive uncertainty explicitly.**

The two replacement directions are:

1. **Direct probabilistic stereo prediction**
   - learn an explicit parametric predictive distribution for disparity;
   - primary candidates: Gaussian heteroscedastic and Laplace heteroscedastic models.

2. **Conditional residual diffusion**
   - retain a strong deterministic mean disparity predictor;
   - learn the conditional residual distribution with a diffusion model.

Both directions assume that the deterministic foundation has first been made as strong and stable as reasonably possible.

---

# 1. Current research position

ReliableEndo-GS separates the project into:

```text
FOUNDATION / BASELINE
├── Stage 1: stereo → disparity
└── Stage 2: deterministic Gaussian reconstruction/rendering

RELIABLEENDO-GS RESEARCH
├── Phase I: reliability / uncertainty
└── Phase II: undecided
```

The current Phase I historically explored reliability signals such as:

```text
update magnitude
population variance
left-right consistency
photometric residual
```

These signals may correlate with prediction error, but they are not themselves guaranteed to be calibrated predictive uncertainty.

Historical evidence already showed the weakness of that approach:

```text
heuristic signal
    ↓
may correlate with error
    ↓
but calibration does not transfer reliably
    ↓
downstream reliability utility becomes weak
```

Therefore, if the new Phase I experiment on full SCARED-C still fails, the next step should **not** be another heuristic proxy.

Instead:

```text
prediction
    ↓
explicit predictive distribution
    ↓
calibration
    ↓
geometry propagation
    ↓
probabilistic reconstruction only if justified
```

---

# 2. Shared principles

## 2.1 Separate mean quality from uncertainty quality

The quality of the mean disparity predictor and the quality of the uncertainty model must be evaluated separately.

Preferred decomposition:

```text
deterministic stereo model
        ↓
mean disparity μ_d
        ↓
uncertainty model
        ↓
predictive distribution around μ_d
```

Whenever possible, keep the deterministic mean fixed while comparing uncertainty alternatives:

```text
same μ_d
├── Gaussian uncertainty
├── Laplace uncertainty
└── diffusion residual distribution
```

This gives clean attribution.

## 2.2 Predictive uncertainty is not automatically total uncertainty

A heteroscedastic head or one diffusion model primarily represents conditional predictive uncertainty. It does **not** automatically provide complete epistemic uncertainty.

If epistemic uncertainty is later needed, treat it separately using methods such as ensembles or other model-level uncertainty techniques.

## 2.3 Calibration is mandatory

Useful uncertainty requires more than:

```text
high uncertainty ↔ high error
```

The stronger requirement is:

```text
predicted uncertainty
≈
observed residual distribution
```

Evaluation must therefore separate:

1. ranking quality;
2. distribution fit;
3. calibration;
4. calibration under sequence shift;
5. downstream geometric utility.

## 2.4 Dataset 6 remains final untouched data

Under the current protocol, `dataset_6` remains the strongest final holdout.

Do not use it for:

- architecture selection;
- uncertainty-head selection;
- loss tuning;
- calibration fitting;
- threshold selection;
- diffusion hyperparameter selection;
- checkpoint selection.

---

# 3. Direction A — Direct Probabilistic Stereo Prediction

## 3.1 Core idea

Instead of producing only:

```text
d_hat
```

the model produces the parameters of:

```text
p(d | I_L, I_R)
```

The simplest form is:

```text
μ_d = predicted disparity mean
s_d = predicted uncertainty scale
```

Two practical candidates:

```text
A1. Gaussian heteroscedastic
A2. Laplace heteroscedastic
```

These form the primary non-heuristic probabilistic baseline.

---

## 3.2 Gaussian heteroscedastic model

For each valid pixel:

```text
d ~ Normal(μ_d, σ_d²)
```

Predict:

```text
μ_d
raw_scale
```

and convert:

```text
σ_d = softplus(raw_scale) + ε
```

The per-pixel Gaussian negative log-likelihood is:

```text
L_G
=
0.5 * ((d_gt - μ_d)² / σ_d²)
+
log σ_d
```

up to constants.

Interpretation:

```text
large error + small σ
→ heavily penalized

large error + larger σ
→ allowed but still penalized

huge σ everywhere
→ penalized through log σ
```

So the network cannot trivially solve the objective by predicting infinite uncertainty.

---

## 3.3 Laplace heteroscedastic model

Stereo residuals may contain heavy tails caused by:

- occlusion;
- specular tissue;
- low texture;
- depth discontinuities;
- correspondence failures.

Use:

```text
d ~ Laplace(μ_d, b_d)
```

with:

```text
b_d = softplus(raw_scale) + ε
```

The per-pixel NLL is:

```text
L_L
=
|d_gt - μ_d| / b_d
+
log b_d
```

Laplace is more tolerant of large outliers and may fit endoscopic stereo residuals better than Gaussian.

---

## 3.4 Preferred architecture

### Option 1 — joint prediction

```text
shared stereo backbone
        ↓
shared recurrent/refinement features
        ├── disparity head → μ_d
        └── uncertainty head → σ_d or b_d
```

Advantage: end-to-end optimization.

Disadvantage: changes mean and uncertainty at the same time, making attribution harder.

### Option 2 — frozen mean + uncertainty head

Preferred first experiment:

```text
frozen deterministic Stage 1
        ↓
μ_d + intermediate stereo features
        ↓
uncertainty head
        ↓
σ_d or b_d
```

Possible inputs:

- left-image features;
- right-image features;
- correlation/matching features;
- recurrent hidden state;
- current disparity `μ_d`.

Do **not** use heuristic uncertainty maps as the main target.

This design provides the cleanest comparison because the deterministic mean is unchanged.

---

## 3.5 Training target

For a frozen mean:

```text
e = d_gt - μ_d
```

Conceptually:

```text
Gaussian:
e ~ Normal(0, σ_d²)

Laplace:
e ~ Laplace(0, b_d)
```

The uncertainty model therefore learns the conditional residual distribution rather than a hand-crafted proxy.

A later extension may predict a residual correction `δμ`, but that should not be part of the first controlled experiment because it mixes mean correction with uncertainty estimation.

---

## 3.6 Training objectives

### Gaussian

```text
L = NLL_Gaussian
```

Useful numerical safeguards:

```text
σ_min
σ_max
log-scale clipping
gradient clipping
```

### Laplace

```text
L = NLL_Laplace
```

with analogous safeguards for `b_d`.

The main objective should remain likelihood-based.

Do not optimize correlation between predicted uncertainty and error as the primary objective.

---

## 3.7 Evaluation

### Mean disparity quality

Always report:

```text
EPE
median AE
signed bias
bad3
bad5
```

This verifies that uncertainty modeling has not damaged the mean.

### Distribution quality

For Gaussian/Laplace:

```text
negative log-likelihood
```

### Calibration

For nominal intervals such as:

```text
50%
80%
90%
95%
```

measure empirical GT coverage.

A calibrated model should approximately satisfy:

```text
nominal 90% interval
→ 90% empirical coverage
```

Coverage must be paired with interval width because arbitrarily wide intervals can achieve high coverage.

Report:

```text
coverage
mean interval width
per-sequence calibration
macro calibration error
```

### Ranking / selective prediction

Secondary diagnostic:

```text
sort pixels by predicted uncertainty
remove the most uncertain fraction
measure remaining EPE
```

Compare against:

```text
random removal
oracle removal using true error
```

This tests ranking usefulness but does not define calibration.

---

## 3.8 Cross-sequence transfer

This is critical for ReliableEndo-GS.

Report calibration and uncertainty quality:

```text
per sequence
macro across sequences
```

Do not rely only on pooled pixels.

The main question is whether uncertainty remains calibrated under sequence shift rather than only on the sequences used to fit the uncertainty head.

---

## 3.9 Post-hoc calibration

If raw uncertainty has useful ranking but imperfect absolute calibration, allow a separate post-hoc stage such as:

```text
global multiplicative scaling
variance/scale temperature
isotonic calibration
```

Fit calibration parameters only on the designated calibration subset.

Report both:

```text
raw uncertainty
calibrated uncertainty
```

Do not use final-holdout data for calibration.

---

## 3.10 Optional conformal extension

If the probabilistic model contains useful predictive information but absolute coverage remains imperfect, a later extension is:

```text
probabilistic model
→ calibration set
→ conformal prediction interval
```

Output:

```text
[d_low, d_high]
```

Conformal prediction should be viewed as a calibration/coverage layer, not as a replacement for the predictive model.

---

## 3.11 Propagation from disparity to depth

Stereo depth:

```text
Z = fB / d
```

The derivative is:

```text
dZ/dd = -fB / d²
```

For small uncertainty:

```text
σ_Z²
≈
(fB / d²)² σ_d²
```

This immediately shows that identical disparity uncertainty can produce very different depth uncertainty depending on disparity.

For larger uncertainty, sampling is preferable:

```text
sample disparity
→ convert each sample to depth
→ estimate depth distribution
```

---

## 3.12 Propagation to 3D

For rectified geometry:

```text
X = (u - c_x) Z / f_x
Y = (v - c_y) Z / f_y
Z = fB / d
```

The predictive disparity distribution induces a 3D positional distribution.

A first-order approximation may use:

```text
Σ_xyz = J Σ Jᵀ
```

where `J` is the Jacobian of the geometric transformation.

Important semantic rule:

> positional uncertainty covariance is not automatically identical to the geometric support covariance of a Gaussian splat.

Keep those concepts separate.

---

## 3.13 Possible use in Stage 2

Only after disparity uncertainty is validated:

```text
uncertain depth
→ positional uncertainty
```

Possible downstream uses:

1. uncertainty-aware Gaussian center covariance;
2. opacity modulation;
3. uncertainty-weighted Gaussian optimization;
4. confidence-aware cross-view fusion.

These are downstream experiments, not evidence that the uncertainty estimator itself is valid.

---

## 3.14 Success gates

### A1 — mean preservation

Mean disparity must remain stable.

### A2 — distribution fit

Gaussian/Laplace must outperform trivial uncertainty baselines such as constant global scale.

### A3 — calibration

Intervals must achieve meaningful coverage without pathological width.

### A4 — transfer

Calibration/ranking must remain useful on held-out sequences.

### A5 — downstream relevance

Only after A1–A4 pass should uncertainty enter Stage 2.

---

## 3.15 Advantages

Direction A is:

```text
simple
fast
interpretable
cheap at inference
easy to calibrate
easy to propagate analytically
```

It should be the first replacement after heuristic Phase I fails.

---

## 3.16 Limitation

Gaussian and Laplace assume the residual distribution is summarized by one center and one scale.

Stereo ambiguity may instead be:

```text
multimodal
skewed
heavy-tailed
spatially correlated
```

That motivates Direction B.

---

# 4. Direction B — Conditional Residual Diffusion

## 4.1 Core idea

Keep the deterministic Stage 1 mean predictor:

```text
μ_d = f_stereo(I_L, I_R)
```

Define:

```text
e* = d_gt - μ_d
```

Then learn:

```text
p(e | I_L, I_R, μ_d, stereo features)
```

with a conditional diffusion model.

At inference:

```text
e^(1), e^(2), ..., e^(K)
~
p(e | condition)
```

and:

```text
d^(k) = μ_d + e^(k)
```

The set of samples approximates the predictive disparity distribution.

---

## 4.2 Why residual diffusion?

Using diffusion to predict full disparity directly would mix:

```text
mean prediction
```

with:

```text
uncertainty modeling
```

Residual diffusion is cleaner because Stage 1 already solves most of the correspondence problem.

The diffusion model learns:

```text
where the deterministic predictor may be wrong
how large the error may be
what residual structures are plausible
```

This also enables a controlled comparison against Gaussian/Laplace with the same frozen mean predictor.

---

## 4.3 Conditioning

Preferred conditioning inputs:

```text
I_L
I_R
μ_d
multiscale stereo features
correlation/matching features
```

The frozen Stage 1 network provides the deterministic mean and useful feature context.

Do not require handcrafted heuristic uncertainty maps.

---

## 4.4 Diffusion objective

Let:

```text
e_0 = e*
```

Forward process:

```text
e_t
=
sqrt(alpha_bar_t) * e_0
+
sqrt(1 - alpha_bar_t) * epsilon
```

where:

```text
epsilon ~ Normal(0, I)
```

The denoiser receives:

```text
e_t
t
condition
```

A standard objective is:

```text
L_diff
=
E ||epsilon - epsilon_theta(e_t, t, condition)||²
```

Alternative parameterizations such as `v` prediction may be tested only if implementation/training stability justifies them.

---

## 4.5 What to diffuse

### B1 — full-resolution residual map

Advantages:

- preserves spatial structure;
- models correlated errors;
- direct disparity samples.

Disadvantages:

- expensive;
- memory intensive.

### B2 — latent residual

```text
residual map
→ encoder
→ residual latent
→ diffusion
→ decoder
```

Advantages:

- cheaper;
- faster.

Disadvantages:

- adds another learned representation;
- can lose local uncertainty detail;
- makes attribution harder.

### B3 — lower-resolution / patch residual model

A compromise:

```text
predict structured residual distribution at lower resolution
→ upsample/refine
```

The first implementation should choose the simplest version that fits the available compute budget.

---

## 4.6 Conditional denoiser architecture

A reasonable design:

```text
Frozen Stage1
     ↓
μ_d + multiscale stereo features
     ↓
condition encoder
     ↓
conditional residual U-Net / denoiser
     ↓
sampled residual e
     ↓
d_sample = μ_d + e
```

Condition injection can use:

```text
concatenation
FiLM
cross-attention
multiscale feature injection
```

Do not begin with the most complex mechanism without evidence it is needed.

---

## 4.7 Sampling

Generate:

```text
K residual samples
```

Typical experimental values may include:

```text
K = 8
K = 16
K = 32
```

depending on compute.

From disparity samples obtain:

```text
sample mean
sample median
sample variance
quantiles
prediction intervals
```

Important:

> Diffusion sample variance is not automatically total uncertainty.

It is the empirical spread of the learned conditional predictive distribution.

---

## 4.8 Preserve the deterministic mean initially

Initial study:

```text
final deterministic prediction = μ_d
```

Use diffusion only to model distribution around it.

A later analysis may compare:

```text
μ_d
```

against:

```text
mean_k(d^(k))
```

but if the diffusion sample mean is used as the point estimate, report this separately because the mean predictor has effectively changed.

---

## 4.9 Why diffusion may be useful

A Gaussian or Laplace model compresses uncertainty into:

```text
one center + one scale
```

Diffusion can represent, in principle:

```text
multimodality
asymmetry
heavy tails
spatial correlation
structured residual maps
```

This may matter in:

- specular regions;
- repetitive texture;
- occlusions;
- low-texture tissue;
- depth boundaries;
- ambiguous correspondences.

The scientific justification for diffusion is therefore **distributional flexibility**, not simply architectural novelty.

---

## 4.10 Prediction intervals from samples

For each pixel estimate quantiles such as:

```text
q_05
q_50
q_95
```

A 90% interval is:

```text
[q_05, q_95]
```

This does not assume symmetry or a fixed parametric form.

Evaluate:

```text
nominal coverage
empirical coverage
interval width
per-sequence calibration
```

Diffusion samples are not automatically calibrated.

---

## 4.11 Sample-based scoring

Exact likelihood may be impractical for diffusion.

Useful alternatives include:

```text
CRPS
energy score
coverage-width tradeoff
```

For scalar disparity, empirical CRPS can be approximated by:

```text
CRPS
≈
(1/K) Σ |x_k - y|
-
(1/(2K²)) ΣΣ |x_i - x_j|
```

where:

```text
x_k = sampled disparity
y   = ground-truth disparity
```

CRPS rewards samples close to GT while penalizing unnecessary spread.

---

## 4.12 Failure modes

### Under-dispersion

```text
samples almost identical
```

The model is too confident.

### Over-dispersion

```text
samples extremely diverse
```

Coverage may look good but intervals become uselessly wide.

### Mean correction masquerading as uncertainty

If diffusion samples systematically shift the mean, the model may be acting as another disparity-refinement network instead of uncertainty modeling.

### Poor transfer

Diffusion may fit training residuals extremely well but fail to preserve calibration on unseen sequences.

All four cases must be tested explicitly.

---

## 4.13 Spatial structure

One advantage over independent per-pixel scale heads is the ability to model correlated residual patterns.

Qualitatively inspect whether sample differences concentrate coherently around:

```text
occlusion boundaries
specular regions
low-texture tissue
depth discontinuities
```

But qualitative samples are supporting evidence only.

The main claim still requires quantitative calibration and transfer.

---

## 4.14 Compute requirements

Diffusion is substantially more expensive than a Gaussian/Laplace head.

Measure:

```text
training GPU-hours
peak VRAM
inference latency
sampling steps
number of samples K
```

Possible later acceleration:

```text
DDIM
few-step samplers
distillation
latent diffusion
consistency models
```

Do not optimize sampling speed before the base diffusion model demonstrates scientific value.

---

## 4.15 Geometry propagation by sampling

This is one of the strongest advantages of Direction B.

For each disparity sample:

```text
d^(k)
```

compute:

```text
Z^(k) = fB / d^(k)
```

then:

```text
X^(k), Y^(k), Z^(k)
```

This yields a sampled 3D positional distribution.

Estimate:

```text
mean 3D position
3×3 positional covariance
depth quantiles
anisotropy
```

This naturally handles nonlinear disparity-to-depth propagation and non-Gaussian distributions.

---

## 4.16 Connection to Gaussian splatting

A possible downstream mapping is:

```text
sampled 3D positions
        ↓
mean 3D position
+
uncertainty covariance
```

However:

```text
uncertainty covariance
```

represents uncertainty about where the 3D point is.

It is not automatically identical to:

```text
Gaussian surface/support covariance
```

A future method must define how these two concepts interact rather than silently treating them as the same quantity.

---

## 4.17 Success gates

### B1 — stable diffusion learning

Residual samples must be finite and non-collapsed.

### B2 — better distribution model

Diffusion should improve sample-based scoring/calibration over simple parametric baselines.

### B3 — calibration

Quantile intervals must achieve useful coverage and sharpness.

### B4 — transfer

Calibration must remain useful across sequence shift.

### B5 — compute justification

The gain over Gaussian/Laplace must justify the larger inference/training cost.

### B6 — geometric utility

Only after B1–B5 pass should sampled uncertainty be integrated into Stage 2.

---

# 5. Controlled comparison

If both directions are activated, use one frozen deterministic mean predictor.

```text
                  ┌── Gaussian head
frozen Stage1 μ ──┼── Laplace head
                  └── residual diffusion
```

Do not compare three different mean stereo models.

That would make uncertainty attribution ambiguous.

---

## 5.1 Shared comparison table

| Property | Gaussian | Laplace | Residual Diffusion |
|---|---|---|---|
| Predictive form | Normal | Laplace | Sample-based |
| Output | μ, σ | μ, b | K residual samples |
| Mean can stay frozen | Yes | Yes | Yes |
| Heavy-tail handling | Limited | Better | Flexible |
| Multimodality | No | No | Yes, in principle |
| Spatial correlation | Limited if pixelwise | Limited if pixelwise | Stronger potential |
| Closed-form NLL | Yes | Yes | Usually no |
| Calibration intervals | Analytic | Analytic | Empirical quantiles |
| Nonlinear geometry propagation | Approx. or sampling | Approx. or sampling | Natural by sampling |
| Inference cost | Low | Low | High |
| Implementation complexity | Low | Low | High |

---

# 6. Recommended fallback order

If current Phase I on full SCARED-C still fails:

```text
Current heuristic Phase I
        ↓ FAIL
        ↓
STOP heuristic search
        ↓
freeze strongest deterministic mean
        ↓
Direction A
Gaussian vs Laplace
        ↓
calibration + transfer evaluation
```

If Direction A succeeds:

```text
use the simpler calibrated probabilistic model
→ propagate uncertainty to geometry
→ test downstream utility
```

If Direction A is clearly insufficient because the residual distribution is too complex:

```text
Direction B
conditional residual diffusion
```

Then compare diffusion against the same frozen mean and same evaluation protocol.

If diffusion does not materially outperform the parametric alternatives, do not use it in the final method.

---

# 7. Recommended research claims

## Direction A

A defensible claim would be:

> ReliableEndo-GS explicitly models per-pixel predictive disparity uncertainty with a calibrated heteroscedastic likelihood and propagates this uncertainty through stereo geometry.

This requires evidence for:

```text
distribution fit
calibration
cross-sequence transfer
geometry utility
```

## Direction B

A defensible claim would be:

> ReliableEndo-GS models the conditional distribution of stereo residuals using residual diffusion, enabling non-Gaussian predictive uncertainty and sample-based propagation into 3D reconstruction.

This requires evidence that diffusion adds value over Gaussian/Laplace baselines.

---

# 8. Claims to avoid

Do not claim:

```text
sample variance = total uncertainty
```

Do not claim:

```text
correlation with error = calibrated uncertainty
```

Do not claim:

```text
uncertainty covariance = Gaussian surface covariance
```

Do not claim:

```text
diffusion samples = epistemic uncertainty
```

Do not claim transfer robustness without held-out transfer evidence.

---

# 9. Practical implementation sequence

```text
1. Finish current Phase I on full SCARED-C.

2. If it passes:
   continue current direction.

3. If it fails:
   stop heuristic reliability work.

4. Freeze the strongest deterministic Stage1 mean predictor.

5. Implement Gaussian heteroscedastic uncertainty.

6. Implement Laplace heteroscedastic uncertainty.

7. Compare:
   - NLL
   - coverage
   - interval width
   - calibration
   - transfer
   - selective prediction
   - compute

8. Select the strongest parametric baseline.

9. If parametric uncertainty is insufficient:
   implement conditional residual diffusion.

10. Compare diffusion against the same frozen mean:
    - CRPS
    - calibration
    - coverage-width tradeoff
    - transfer
    - compute

11. Calibrate the selected predictive model.

12. Propagate:
    disparity distribution
    → depth distribution
    → 3D positional uncertainty.

13. Only after uncertainty semantics are validated:
    integrate uncertainty into deterministic Stage2.

14. Measure whether it improves reconstruction reliability.

15. Access the final untouched dataset only after all choices are frozen.
```

---

# 10. Final decision tree

```text
Phase I on full SCARED-C
              │
              ├── PASS
              │     └── continue current Phase I
              │
              └── FAIL
                    │
                    ▼
            STOP heuristics
                    │
                    ▼
       freeze strong deterministic μ_d
                    │
                    ▼
       DIRECT PROBABILISTIC MODEL
          ├── Gaussian
          └── Laplace
                    │
          calibration + transfer
                    │
          ┌─────────┴─────────┐
          │                   │
        PASS             INSUFFICIENT
          │                   │
          ▼                   ▼
geometry propagation   CONDITIONAL RESIDUAL
                           DIFFUSION
                              │
                     sample distribution
                              │
                     calibration + transfer
                              │
          └───────────┬───────┘
                      ▼
           disparity distribution
                      ↓
             depth distribution
                      ↓
          3D positional uncertainty
                      ↓
        Stage2 uncertainty integration
                      ↓
         downstream utility evaluation
```

---

# 11. Core research principle

The central change is:

```text
OLD
heuristic proxy
→ correlation with error
→ try to use it as reliability

NEW
learn predictive distribution directly
→ verify calibration
→ verify sequence transfer
→ propagate only validated uncertainty
```

If the current full-SCARED-C Phase I still fails, ReliableEndo-GS should move toward **explicit distribution learning**, not another handcrafted reliability proxy.
