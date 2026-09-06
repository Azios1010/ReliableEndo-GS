# Phase-I untouched final evaluation data

This record freezes `dataset_5/keyframe_1` and `dataset_8/keyframe_2` as the
untouched final evaluation set for the preregistered Phase-I protocol.

## Frozen identity

- External manifest: `E:\datasets\reliable-endo-gs\scared\manifests\scared_phase1_final_untouched_v1.json`
- External manifest SHA-256: `D9EA9A82439BAA3B52EEB922AC4A30F5BC3305C3A84C09CA080FF33CADCC3D28`
- Source split: `splits/scared/phase1_final_untouched_v1.json`
- Source split SHA-256: `D13031A5E269B99D13F889AEA4BCCB0F0E35D9F974D5CE689FE426A3E09A72E2`
- Data config: `configs/data/scared_phase1_final_v1.yaml`
- Data config SHA-256: `0CFC3E40DCE1F7B42EBA90485782D683FC7855F103EB37DA9076960A9C7E5DC2`
- Phase-I protocol SHA-256: `E4FCAF8A15F0DBA00294DE02AAA74ED12BEC095AA797D97022079C216D663032`
- Gate-0 protocol SHA-256: `DAE773A0747F806D43D1C5B1BE147C69308B4D1CE4463FE4721F0C3EAA411315`
- Deterministic Stage-2 baseline SHA-256: `32D42DE10A4C683C31FB4B3E0DE831F900E11197CDF1769B18CF3BBAFA5EE16F`
- Endo-E2E-GS SHA: `186fa2b4a2159b28393492f6df1aa444b54391a8`

The materialization source commit was
`fffc1881d0abd52101c235771883f920a5b0b045`; the post-freeze commit is
recorded in Git history. The preregistered Phase-I file retains its historical
registration provenance and hash; no protocol content was edited during data
materialization.

The external manifest contains all processed frame IDs and all canonical-loader
selected sample IDs. The frozen loader policy is `phase=all`, `skip_every=2`
for both final sequences, deterministic sequence/frame order, no shuffle, no
augmentation, and no random subsampling. This yields 99 samples from
`dataset_5/keyframe_1` and 347 samples from `dataset_8/keyframe_2` (446 total).

## Isolation policy

These sequences are not available for uncertainty-proxy selection, proxy
combination, calibration fitting or threshold selection, geometry-uncertainty
design, probabilistic-Gaussian design, opacity/mass-rule selection, cross-view
design, architecture or hyperparameter selection, checkpoint selection, early
stopping, ablation selection, or qualitative development iteration.

They may be opened for model evaluation only after the Phase-I architecture,
hyperparameters, ablations, and checkpoint-selection rule are frozen.

Development remains on `dataset_1/keyframe_1`, `dataset_2/keyframe_1`,
`dataset_3/keyframe_1`, with `dataset_7/keyframe_2` and
`dataset_4/keyframe_4` as development holdouts. Those development holdouts
have already influenced prior checkpoint selection and Gate-0 reasoning and
are not pristine final tests.

`dataset_9/keyframe_3` remains excluded because the prior data-only audit
reproduced substantial zero-valid disparity frames; it is not substituted into
this final set.

## Processing identity

Payloads were acquired from the canonical SCARED Hugging Face mirror and
processed by the existing v3 pipeline. The canonical preprocessing script
SHA-256 is `ABE34F324870CE8DDEB8729FFD796285106B4B810576D80FFC3BB3D526D45AB6`.
The preprocessing contract is unchanged: video top half is left, bottom half is
right; OpenCV stereo rectification/remapping is used; `intr0=P1[:3,:3]`,
`intr1=P2[:3,:3]`, and `extr0=R1_homogeneous @ camera_pose`; disparity is
positive left-reference pixel disparity with invalid value zero.

No model was instantiated or run on either final sequence during materialization
or this audit.
