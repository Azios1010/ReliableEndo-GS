# Signed expected gain in the guiding harness

## Scope

The user approved factorizing gain into probability of positive utility,
conditional positive benefit and conditional nonpositive cost. Joint remains
the primary experimental path; frozen student is a separate comparison. The
physical utility, local/global/any risk, cutoffs, budgets and SCARED roles remain
the same. This implementation does not certify a research gate or real gain.

## Meaning and architecture

For each stereo/Gaussian action and pre-action masked feature vector x:

```text
p = P(U > 0 | x)
g_plus  = E[U  | U > 0, x]
g_minus = E[-U | U <= 0, x]
expected_gain = p*g_plus - (1-p)*g_minus
```

`routing/gain.py` owns decoding, this expectation and component supervision.
The magnitude outputs use softplus and the same TRAIN-fitted utility scale for
both actions. Expected utility can be negative. STOP remains exact zero gain
and risk with no learned parameters. The two independent gain MLPs have four
readouts: positive logit, positive magnitude, nonpositive magnitude and the
existing local geometry auxiliary. Physical risk MLPs remain independent.

Positive utility is **not** equivalent to physical safety. A render improvement
may have positive U and violate a local guard. Risk retains the actual local/
global/any labels. Selection uses expected gain after the existing risk and
budget constraints. Rare beneficial stereo outcomes are not rewarded solely
for being rare; benefit probability and downside both matter.

The opt-in settings are `selector_gain_mode: signed_expectation` and
`gain_probability_weight: 1.0`. They require physical balanced v2 supervision,
independent action heads and per-action balancing. Existing configurations
default to historical regression, omit the new neutral fields from their
serialized identities and retain the old network shape.

## Guiding harness and loss

The same `BalancedHarness.collect` executes the real actions and reports
canonical signed utility and harm from separate offline truth. No private
repair or alternative score is introduced. Only complete observed regions
reach selector learning; unknown outcomes never become negative/safe labels.

- Sign: unweighted binary cross entropy on U>0, preserving the observed row
  distribution. It is outside the magnitude/local gradient balancer so the
  magnitude EMA cannot silently suppress sign supervision.
- Magnitudes: mean squared error on each sign class, in common normalized
  utility units. Each class uses its own count/weight denominator. The mean
  squared loss targets conditional means; a robust median estimate would not
  in general satisfy the expectation formula. The fixed class average is
  0.5*(positive MSE + nonpositive MSE). An absent class contributes differentiable
  zero and receives no invented magnitude label. U=0 is nonpositive, cost0.
- Existing safe-pair ranking uses expected utility. The local auxiliary and
  physical risk losses keep their existing meaning and ownership.

Conditional magnitude losses can preserve the importance of rare large
benefits while sign probability reflects frequency. This does not guarantee
that available input features predict the outcomes. Repeated single-region
updates and evolving actor features remain limitations of the online pilot.
Changing class sampling requires accounting for sampling probabilities before
interpreting sign outputs as probabilities on the original population.

## Diagnostics and compatibility

Training and native evaluation record probability, conditional positive gain,
conditional negative cost and expected gain in [regions, STOP/ST/GS] order.
Sign BCE/Brier/rate and class counts are logged per action. The fixed-feature
head probe scores sign and conditional magnitude errors.

Native gain diagnostics exclude unknown candidates, distinguish positive from
positive-and-safe outcomes, and report recovery of safe stereo/Gaussian oracle
opportunities, using the same selected region. Utility, physical harm and
render/depth results remain necessary. Selecting three types is not acceptance.

New joint/frozen configurations:

- `configs/routing/intervention_metric_signed_gain.yaml`
- `configs/routing/intervention_metric_signed_gain_frozen_student.yaml`

They differ only in student update mode. Use the common
`configs/training/student_joint_routing.yaml` and explicitly bound trials.
Strict resume rejects historical head topology before mutation. A new run may
use `--initialize-model-from` to warmstart model weights with fresh repairs,
gain/risk heads and Adam; this is not full-state resume. Keep old checkpoints.
Historical scalar gain/risk calibration is incompatible and fails early. New
component-aware probability/calibration fitting is not implemented here.

## Evidence limits

Synthetic evidence must cover signed units, packing, natural-frequency sign
learning, conditional class denominators, missing classes, rare large benefits,
detached inputs, ownership, frozen comparison, resume and unknown exclusion.
Native smoke is a runtime check. Neither synthetic success nor native smoke
proves improved action selection on SCARED. The earlier trial's skewed fold
coverage, rare global-harm positives and token-normalization drift remain
separate issues; no data roles, evaluation metrics or recorded evidence were
changed to hide them.

## Bounded native evidence (2026-10-13)

The final implementation passed 264 focused CPU tests, mypy on 232 source
files, Ruff and formatting checks. Native GPU0 smoke updated student,
reconstruction head, both repairs and selector; zero repairs reproduced both
views exactly. A separate joint trial completed 256 seeded FIT samples with
256 model/action updates and 612 selector updates. No teacher inference ran.

The comparison used the previous regression trial's initialization and image
order, followed by the same 28 development images. Initial model, actions,
risk weights and CPU/CUDA RNG were identical. Trained actors and candidate
outcomes differ: this is an end-to-end joint comparison, not an isolated
selector ablation. The backbone's earlier validation/calibration ancestry
overlaps this cohort; these are development diagnostics, not final-test results.

| Development result | Previous regression | Signed expectation |
| --- | ---: | ---: |
| Selector STOP / stereo / Gaussian | 0 / 0 / 28 | 0 / 14 / 14 |
| Oracle STOP / stereo / Gaussian | 4 / 9 / 15 | 3 / 11 / 14 |
| Mean selected physical utility | 0.102219 | 0.099719 |
| Selected physical guard violations | 2 | 4 |
| Selected unknown metric outcomes | 1 | 1 |
| Oracle agreement | 15/28 | 15/28 |
| Mean safe gain regret | 0.100723 | 0.085786 |

The new selector captured 7/11 safe stereo oracle opportunities. However,
selected utility did not improve and guard violations increased; selecting
both repairs does not establish a successful routing contribution. On the new
actor, Gaussian-only selection achieved mean utility 0.103727 with 3 guard
violations, so the selector did not outperform this simpler comparison.

Read-only replay found no training/evaluation label or head/decision mismatches.
Expected-gain component reconstruction differed by at most 6.82e-9. All known
stereo/Gaussian candidates still had positive predicted expected gain. Stereo
mean positive probability was 0.684 versus observed positive rate 0.500;
its sign AUROC was 0.590. Any-harm AUROC was 0.459 for stereo and 0.518 for
Gaussian on 32 known candidates each. The three selected harmful stereo
cases had predicted harm probabilities 0.086, 0.095 and 0.151; their actual
utility was negative. This points to remaining gain/sign and risk prediction
errors, rather than a transform/packing error or a need to force action quotas.

The implementation is runtime-tested; routing quality remains unproven.
Further validation should separate actor changes from head fitting, monitor
feature normalization drift, and assess sign/risk discrimination on more
representative observed regions. Component-aware calibration and native
frozen-student comparison were not run. Historical scalar calibration remains
incompatible with this mode. No thresholds or utility weights were tuned on
these 28 images, and no full-data or final-test run was launched.
