# Discovery Gate 0: Frozen Evidence

Status: **FROZEN**

This record preserves the bounded Discovery Gate 0 evidence. It does not
modify the immutable Gate-0 preregistration and does not authorize Phase-I or
RiskRoute implementation.

## Identity and provenance

| field | value |
| --- | --- |
| artifact | `GPS-GS_stage2_deterministic_v3_best_step003000.pth` |
| step | `3000` |
| checkpoint SHA-256 | `32D42DE10A4C683C31FB4B3E0DE831F900E11197CDF1769B18CF3BBAFA5EE16F` |
| ReliableEndo-GS commit | `a2cd8a71aa6f4e8f9962bbea64149a72a5de0281` |
| Endo-E2E-GS commit | `186fa2b4a2159b28393492f6df1aa444b54391a8` |
| Stage-1 checkpoint SHA-256 | `202ECC041B61B720408E0548319A1F9385C07DB784B824751482C615A0F46964` |
| manifest SHA-256 | `67C4B16BDAEC11B653A5AD757897F4699D37F8C74F4B320E084E5A5492665786` |
| split SHA-256 | `A03495E8C345C3396463D6E7F9A85F625699A3504804B71075929E67B2605590` |
| v3 config SHA-256 | `933E9A2777B9E0080CC00ACAC3E476DA2B8EA2F1142CC036E671195E24940A28` |
| immutable protocol SHA-256 | `DAE773A0747F806D43D1C5B1BE147C69308B4D1CE4463FE4721F0C3EAA411315` |

The original and frozen baseline files were byte-identical, CPU-loadable, and
contained finite model, optimizer, and scheduler state at step 3000.

The immutable result artifacts are retained outside the repository. Their
hashes are recorded in `configs/research/gate0.yaml`:

- deterministic selection: 10 frames across five sequences;
- H1: 40 perturbation cases;
- H2: 160 fixed-grid regions and 480 action evaluations.

The final synced W&B record is `discovery-gate0-compensation-repairability-final`
(`j1xsrw98`) in group `discovery-gate0`; remote sync finished successfully.

## Frozen protocol

The pre-registered protocol remains unchanged:

- sequences: `dataset_1/keyframe_1`, `dataset_2/keyframe_1`,
  `dataset_3/keyframe_1`, `dataset_7/keyframe_2`, and `dataset_4/keyframe_4`;
- two deterministic strata per sequence: moderate and hard;
- disparity perturbations: `-25%`, `-10%`, `+10%`, `+25%`;
- H2 regionization: fixed 4x4 grid;
- actions: `STOP`, `STEREO_REPAIR`, `GAUSSIAN_REPAIR`;
- no router, budget allocator, Phase-I implementation, or Stage-2 retraining.

The historical preregistration is intentionally not rewritten after observing
the results.

## Finding A: uncertainty is not repairability

For the primary RAFT iteration-disagreement proxy:

| relationship | Spearman rho |
| --- | ---: |
| uncertainty vs actual disparity error | `+0.345409` |
| uncertainty vs stereo repair gain | `-0.384885` |
| uncertainty vs Gaussian repair gain | `-0.067938` |

Uncertainty moderately ranked disparity error, but did not rank intervention
utility in the same way. This is **SUPPORTED / MODERATE** evidence for an
uncertainty–repairability mismatch, not a universal law.

## Finding B: limited Gaussian-side compensation

Gaussian-only adaptation improved same-view RGB L1 by approximately `0.004400`
on average. The sequence-cluster bootstrap 95% interval was approximately
`[0.003926, 0.004989]`, and the direction was consistent across all five
sequences.

The strong cross-view concealment hypothesis was not supported. Cross-view
Gaussian compensation gain was approximately `0.000001835` L1, while oracle
stereo center repair improved cross-view L1 by approximately `0.055570`.
Perturbation-induced cross-view degradation was not systematically observed.

Classification: **WEAK / PARTIAL** Gaussian compensation signal;
**INCONCLUSIVE** strong cross-view compensation discovery.

## Finding C: action specialization is not supported

| best action | count | fraction |
| --- | ---: | ---: |
| STOP | 11 | 6.875% |
| STEREO_REPAIR | 0 | 0% |
| GAUSSIAN_REPAIR | 149 | 93.125% |

Gaussian repair dominated stereo repair in `155/160` regions. This shows
action-dependent utility in this intervention design, but not heterogeneous
action specialization. RiskRoute is therefore **PARKED**.

Mean regional gains, where positive means reduced same-view RGB L1, were
`-0.101375` for stereo repair and `+0.005078` for Gaussian repair. Sequence-
cluster bootstrap 95% intervals were `[-0.158043, -0.053900]` and
`[0.004329, 0.005955]`, respectively. Frame-cluster intervals were
`[-0.149009, -0.061061]` and `[0.004381, 0.005738]`.

Mean measured intervention costs were `0.002490 s` for stereo repair and
`0.030553 s` for Gaussian repair. The immutable protocol did not define a
budget-to-action mapping, so these costs do not support a post-hoc frontier.

## Gate-0 decisions

| decision | result |
| --- | --- |
| uncertainty–repairability mismatch | supported, moderate |
| Gaussian compensation | inconclusive as a strong phenomenon |
| action specialization | not supported |
| compute-aware selective repair | inconclusive; no frozen budget mapping |
| roadmap | C: uncertainty representation without RiskRoute |

## Scientific boundary

Gate 0 supports the narrower statement that prediction uncertainty, prediction
error, and intervention utility are distinct quantities in this bounded study.
It does not establish universal uncertainty behavior, a cross-view compensation
law, or a useful repair router.
