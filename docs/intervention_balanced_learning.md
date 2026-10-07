# Balanced local/global action learning

## Scope and retained evidence

The user approved these corrections on 2026-10-06 after the loss/feature audit.
`intervention_selection.v2` is an opt-in joint experiment. Student, Gaussian
head, repairs and selector learn; the teacher stays an external cache source.
This does not certify Gate I/II or change baseline or historical v1 artifacts.
Use original SCARED train folds 1–7; folds 8–9 stay final test. The existing
5,738-frame training manifest and disjoint 112-frame development manifest are
unchanged. Development IDs share keyframes with training, so this is not a
sequence-held-out claim.

The preceding native 256-update v1 pilot selected stereo on all 112 development
frames although its proxy oracle chose Gaussian on 68. Mean predicted versus
observed gain was 0.0984/0.0555 for stereo and 0.00884/0.0607 for Gaussian.
Both actions were budget-feasible in all 448 regions. Depth MAE was
10.576/10.633/10.554 mm for base/selector/oracle; proxy improvement did not
establish metric geometry improvement. These negative results remain evidence.

## Loss scale and parameter ownership

The audit measured final hidden-activation gradients, not full parameter
gradients: harm 0.077698, gain 0.000807, ranking 0.000543. This supported testing
separate encoders and actual parameter-gradient diagnostics, rather than fixing
a multiplier from scalar loss magnitudes alone.

The v2 selector has independent gain and harm MLPs, each 61 → 32 → 32 with ReLU.
Gain has four outputs (global/local utility for both repairs); harm has two.
There are 6,278 trainable parameters with the default width. STOP utility and
harm are fixed zero. Harm BCE and its gradient clipping cannot alter the gain
encoder. Global/local SmoothL1 and ranking losses share the gain encoder.

`training/gradient_balance.py` measures actual shared parameter gradients:

- Selector: gain encoder parameters before the output layer, every joint update.
- Model: stereo feature encoder parameters, every 25 optimizer positions.
- Detached weights use the median active EMA norm divided by each norm, bounded
  to [0.05, 20]. Zero-gradient terms are not amplified. EMA decay is 0.9.
- Logs include component norms, gradient cosines, weights, owner update norms
  and raw losses. Configured priorities are applied after balancing.

These choices are pilot hyperparameters, not established optimal scales.
KD remains screened teacher supervision; accepted targets can still be wrong.

## Local and global supervision

Every eligible grid region receives STOP/stereo/Gaussian interventions from one
pre-update state using the same action registry as inference. With the default
2×2 grid this is four region tables per joint update. Labels are detached;
repairs are trained on their actual local cores/halos, with global preservation
and outside-region damage penalties. They still render full frames.

Let Δ denote before minus after. The canonical component implementation in
`training/quality_components.py` defines global utility as:

```text
G = Δglobal_RGB / 0.07 + Δglobal_warp / 0.1
    - max(0, Δoutside_RGB_damage) / 0.07
    - max(0, Δoutside_warp_damage) / 0.1
L = Δlocal_RGB / 0.07 + Δlocal_warp / 0.1
```

Here each outside damage is after minus before. `L` is separate auxiliary
supervision; it is not added again to global allocation utility. Harm marks
degradation of any declared component beyond the configured normalized
tolerance. Local right RGB uses the fixed pre-action projection of the left
core, rather than the same image rectangle in both cameras. Empty right
support skips that view explicitly; it is never fabricated. Correspondence
support stays fixed so repairs cannot win by removing difficult pixels.

Warp is a photometric proxy, not metric GT. GT remains offline evaluation only;
neither deployment batches, selector features nor normalization consume it.

## Features and frozen normalization

The schema exposes 61 detached features: existing diagnostics plus global RGB,
warp/support, disparity statistics, metric depth statistics, camera/grid
quantities, log scales/anisotropy, density, normals, per-Gaussian native
visibility/radii when available, cost-volume entropy/margin/feature energy and
2×2 spatial error summaries. Availability flags distinguish missing evidence.
There is no invented calibrated confidence, covariance or instrument mask.
The optional matching-evidence pass recomputes stereo features/correlation;
its cost is included in state construction. It is not free.

Normalization fits only the first 32 successful joint training snapshots.
Before fitting, features are raw and gain scale is the configured 0.05 fallback.
After fitting, features use median and IQR/1.349, with a 1e-6 scale floor and
clipping at ±5. Constant features become zero; availability flags stay binary.
This deliberately changes the input domain after warmup. It requires native
learning validation; it is not an already calibrated distribution.

Global and local targets each use one common q75 absolute scale across both
repair actions, floored at 0.01. No mean subtraction changes STOP or gain sign.
Inference decodes utility back to raw units before budget allocation.
Normalization buffers, contributing sample identity chain, gradient EMA and
optimizer states are checkpointed. Inference/evaluation cannot fit or mutate
them and selected deployment rejects an unfitted normalizer. Model-only warm
start from v1 is a fresh experiment; v1 selector/optimizer state is not reused.

## Validation and bounded next run

Synthetic tests cover masks, common units, constant features, training-only
fit/freeze, loss/gradient isolation, all-region labels, owner updates, resume
and corruption guards. A CPU analytic renderer substitutes for CUDA only in
tests. Synthetic learning demonstrates code behavior, not clinical quality.

The next native run must pass the smoke before pilot training. Export the
existing external data/manifest/cache/runtime paths used by the prior pilot,
then set:

```bash
export RELIABLE_ENDO_INTERVENTION_CONFIG=configs/routing/intervention_balanced.yaml
export RELIABLE_ENDO_SMOKE_INTERVENTION_CONFIG=configs/routing/intervention_balanced_smoke.yaml
export RELIABLE_ENDO_SMOKE_CONFIG=configs/training/intervention_balanced_smoke.yaml
bash scripts/run_intervention_pilot.sh 256
```

The script pins physical GPU 0, creates a unique external output directory,
checks 64 cached targets, performs one student and two joint smoke updates,
then up to 256 joint updates and 112-frame development evaluation. The smoke
uses a one-snapshot normalizer only to validate deployment mechanics; the pilot
fits 32 snapshots. It does not generate teacher cache or modify old artifacts.
Native VRAM/runtime remain unverified: eight local repair graphs and gradient
probes can cost more than v1. No ROI-proportional latency or FPS is claimed.
Do not launch a long run until this bounded native check passes and independent
GT, oracle diversity, harm, fixed-action and matched-cost results are reviewed.

## Explicit stereo priority ablation

`stereo_weight` weights the global/local warp gains, repair warp objectives and
outside warp damage. The base warp term enters gradient balancing with that
weight, so inverse-norm balancing can absorb much of its amplitude.
`stereo_training_priority` is a separate factor applied to the base warp
objective **after** balancing. It never enters the norm/EMA probe. Gradient
clipping and Adam still control the eventual parameter update; priority ×3
does not promise a parameter change exactly three times larger.

Neutral priority 1 is omitted from serialized v2 settings to preserve existing
v2 config identities. Non-neutral priority is included in config/checkpoint
identity; changing it requires a fresh experiment, not a resume with different
losses. Historical v1 serialization is preserved.

`configs/routing/intervention_stereo_priority.yaml` and its smoke counterpart
set utility stereo weight 3 and post-balance base warp priority 3. They leave
the weight-1 configs intact. Use the same model-only warm-start checkpoint,
manifest, seed, cache and update bound as the weight-1 pilot to compare learning.
The two controls change together in this pilot; separate ablations are needed
to attribute any improvement to one control. KD weight remains unchanged.

The CPU `selection weight-sweep` command scores fixed interventions already
recorded in a pilot, using the canonical utility and harm implementation:

```bash
python -m reliable_endo_gs.cli.selection weight-sweep \
  --pilot-dir "$RELIABLE_ENDO_RECORDED_PILOT" \
  --output-dir "$RELIABLE_ENDO_SWEEP_OUTPUT" --weights 1 3 5
```

The 256-update/112-frame native pilot recorded on 2026-10-06 gives:

| Stereo weight | Training regions with safe stereo winner / 1,024 | Development recorded-candidate safe stereo winner / 112 |
| --- | ---: | ---: |
| 1 | 128 | 0 |
| 3 | 131 | 0 |
| 5 | 136 | 1 |

Training tables cover all eligible regions at different training snapshots.
Development logs contain only policy-selected candidates, so these counts are
**restricted counterfactuals**, not a full oracle for weights 3/5. This analysis
does not train, render, reload GT, reinterpret old selector predictions or
mutate prior outputs. It does not establish that weighting fixes stereo repair.
The next native smoke/pilot must test new learning and independent GT.
