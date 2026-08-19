# Endo-E2E-GS pinned API and convention audit

## Audit identity and outcome

This record was produced for Plan 01 from the official public repository:

| Field | Observed value |
| --- | --- |
| Project | `Intelligent-Imaging-Center/Endo-E2E-GS` |
| Origin | `https://github.com/Intelligent-Imaging-Center/Endo-E2E-GS.git` |
| Default branch at inspection | `main` |
| Selected immutable commit | `186fa2b4a2159b28393492f6df1aa444b54391a8` |
| Selected commit date/message | 2026-03-16, `Add files via upload` |
| History at inspection | 7 commits, from `6a5d1c8` to the selected commit |
| Integration | Direct Git submodule, unpatched |
| Repository license | MIT, copyright 2025 Intelligent-Imaging-Center |

The official source, README, environment, entry points, and model path were
inspected. The adapter boundary is implementable without patching upstream.
Plan 01 nevertheless remains **IN PROGRESS** because no official checkpoint or
legal inference fixture is published, checkpoint terms are unspecified,
embedded Graphdeco notices are not resolved by an included `LICENSE.md`, and
the required CUDA/compiled environment is unavailable and internally
incomplete on the current machine.

## Repository and entry points

| Upstream path | Symbol/entry point | Scientific role | Adapter exposure | Notes |
| --- | --- | --- | --- | --- |
| `core/raft_stereo.py` | `RAFTStereo`, `FlowUpdateModule` | Recurrent stereo disparity | Final disparity; iterations only when called outside official test mode | Test mode discards intermediate iterations. |
| `core/extractor.py` | `UnetExtractor`, `MultiBasicEncoder` | Multi-scale image/context features | No public feature contract in Plan 01 | `StereoEndoModel` passes encoder features privately to GSRegresser. |
| `lib/network.py` | `StereoEndoModel` | Composite stereo and Gaussian model | Composite output mapping | Mutates nested `data['lmain']`. |
| `lib/utils.py` | `disp2depth` | Upstream disparity-to-depth behavior | Document only | Computes `disp_const / flow_pred`, masks, then batch-global min-max normalization. It is not retained as metric depth. |
| `lib/utils.py` | `depth2pc` | Pixel-center backprojection and world transform | Result maps from `lmain.xyz` | Uses pixel coordinates `0.5 ... W-0.5`, `0.5 ... H-0.5`. |
| `lib/gs_parm_network.py` | `GSRegresser` | Rotation, scale, opacity maps | Maps to `GaussianField` | Rotation normalized over 4 channels; scale Softplus then max-clamped to 0.01; opacity sigmoid. |
| `lib/network.py` | `flow2gsparms` | Gaussian center/attribute assembly | Maps actual dictionary fields | Produces `xyz`, `pts_valid`, `rot_maps`, `scale_maps`, `opacity_maps`. |
| `lib/GaussianRender.py` | `pts2render` | Valid-point selection and RGB preparation | Layout/color conversion mirrored exactly | Flattens image maps, filters by `pts_valid`, converts RGB with `x * 0.5 + 0.5`. |
| `gaussian_renderer/__init__.py` | `render` | CUDA Gaussian rasterization | RGB maps to `RenderOutput` | Hardcodes CUDA tensors; returns only rendered image; consumes rotation/scale, not precomputed covariance. |
| `lib/endo_loader.py` | `SCARED_Dataset` | Upstream data and camera dictionary | Inspection only; Plan 02 owns loading | Left view carries raster matrices/extrinsics; right view carries only image and intrinsics. |
| `train_stage1.py` | `Trainer` | Stereo/depth pretraining | None | Hardcoded SCARED path and CUDA; includes an undeclared TensorFlow import. |
| `train_stage2.py` | `Trainer` | Full stereo + Gaussian rendering training | None | Loads stage-1 non-strictly or stage-2 strictly. |
| `render.py` | `StereoRender` | Official render/inference CLI | Future native smoke/parity path | Source currently instantiates `EndoNeRF_Dataset`; SCARED lines are commented despite README's SCARED command. |

## Observed execution flow

The unmodified composite inference path is:

```text
lmain.img + rmain.img
  -> UnetExtractor
  -> RAFTStereo(..., test_mode=True)
  -> lmain.flow_pred
  -> disp2depth
  -> depth2pc -> lmain.xyz
  -> GSRegresser -> rot_maps, scale_maps, opacity_maps
  -> pts2render valid selection and RGB conversion
  -> diff_gaussian_rasterization -> lmain.img_pred
```

Stage 1 creates `StereoEndoModel(..., with_gs_render=False)` and optimizes the
RAFT sequence loss. Stage 2 creates it with Gaussian regression enabled and
combines disparity loss, RGB L1, and SSIM. Both scripts save a mapping with
`network`, `optimizer`, `scheduler`, and `total_steps`. `render.py` requires
`checkpoint['network']` and loads it strictly on CUDA.

## Contract mappings and transformations

| Upstream field | ReliableEndo-GS field | Shape transformation | Dtype/device | Numerical transformation |
| --- | --- | --- | --- | --- |
| `lmain.flow_pred` | `StereoPrediction.disparity` | None, `[B,1,H,W]` | Preserved | None |
| `lmain.mask` | `StereoPrediction.valid_mask` | None | Device preserved; numeric to bool | `mask >= 0.5`, identical to upstream evaluation |
| directly captured RAFT list | `disparity_iterations` | Sequence preserved | Preserved | None; absent in official test mode |
| `lmain.xyz` | `GaussianField.means3d` | None, `[B,N,3]` | Preserved | None |
| `lmain.img` | `GaussianField.colors` | BCHW to BNC | Preserved | `x * 0.5 + 0.5`, identical to `pts2render` |
| `rot_maps` | `GaussianField.rotations` | BCHW to BNC | Preserved | None |
| `scale_maps` | `GaussianField.scales` | BCHW to BNC | Preserved | None |
| `opacity_maps` | `GaussianField.opacities` | BCHW to BNC | Preserved | None |
| `pts_valid` | `GaussianField.valid_mask` | None, `[B,N]` | Must already be bool | None; invalid points remain represented and masked |
| `lmain.img_pred` | `RenderOutput.image` | None | Preserved | None |

No covariance field is populated. The renderer exposes no rendered depth or
visibility, so those fields remain absent. The adapter does not resize,
normalize images differently, clamp disparity, change opacity, transform scale
parameterization, or recompute covariance.

## Camera and coordinate observations

- The SCARED loader normalizes images from byte RGB to `[-1, 1]` without
  resizing in `__getitem__`.
- `flow_pred` is the single horizontal component of RAFT's `coords1 - coords0`
  update and is used as the denominator of `f*b/d`. The source does not
  independently document disparity sign or rectification assumptions.
- `disp_const` comes from SCARED's reprojection matrix as `Q[2,3] * 1/Q[3,2]`.
  It is not derivable from the current `StereoBatch` without introducing a
  convention not yet owned by Plan 01.
- `depth2pc` treats the stored 3x4 `extr` as a world-to-camera transform and
  returns world-space points using its transpose/inverse algebra.
- Rasterization uses transposed `world_view_transform` and projection matrices
  created in the loader. Only the left view contains the complete required
  fields.
- Quaternion component ordering is not declared by upstream. The map is passed
  unchanged to `diff-gaussian-rasterization`; the adapter therefore records no
  unverified ordering claim.
- Scale is a positive direct scale after Softplus and a maximum clamp, while
  opacity is a sigmoid probability rather than a logit.

These observations are sufficient for lossless output translation, but not to
construct the full official input dictionary from `StereoBatch` alone. Plan 02
must independently verify SCARED calibration, rectification, units, right-camera
pose, and `disp_const` before that bridge is enabled.

## Dependency audit

The committed upstream `environment.yml` pins Python 3.10.13, PyTorch 2.0.1,
torchvision/torchaudio 0.15.2, CUDA 11.8, Taichi 1.5.0, YACS, OpenCV, SciPy,
tqdm, and TensorBoard. The README separately instructs compiling Graphdeco's
`diff-gaussian-rasterization` from another repository without pinning its
commit.

Actual imports add undeclared requirements or risks: `fpsample`, `imageio`,
Pillow, matplotlib, scikit-image, `einops`, `jaxtyping`, `plyfile`, and a
TensorFlow module in `train_stage1.py`. The default `corr_implementation` is
`reg_cuda`, but `corr_sampler` is imported opportunistically and is neither
declared nor built by the repository. Switching to the pure PyTorch `reg`
backend may affect numerical parity and is not silently done by the adapter.

ReliableEndo-GS core requires PyTorch 2.2 or newer and validates on CPU. The
upstream's exact PyTorch 2.0.1 stack is therefore kept as a separate, currently
unverified execution concern; its environment is not merged into core and CPU
CI never imports its CUDA extension.

## License and checkpoint findings

The official root MIT license is preserved. Two copied Graphdeco-derived files
carry a narrower header pointing to a missing `LICENSE.md`, and the compiled
rasterizer is acquired from a separate repository. This is a concrete license
clarification blocker, not a legal conclusion.

The README shows `Endo-E2E-GS_stage2_final.pth` as an argument but provides no
download location, immutable identity, hash, or weight license. Consequently:

- no checkpoint is committed;
- committed baseline config leaves checkpoint fields null;
- checkpoint loading requires an explicit file and expected SHA-256;
- no native inference result or baseline metric is claimed.

## Scientific intervention-point inventory

| Evidence | Upstream location | Accessible without patch? | Contract mapping | Future owner |
| --- | --- | --- | --- | --- |
| Final disparity | `lmain.flow_pred` in `StereoEndoModel.forward` | Yes | `StereoPrediction.disparity` | Plans 03-04 |
| Disparity iterations | list returned by `RAFTStereo` in training mode | Only with a wrapper/direct call; not official test composite | Optional `disparity_iterations` | Plan 04 |
| Depth | `lmain.depth_pred` from `disp2depth` | Yes, but normalized rather than metric | Not promoted as metric depth | Plans 02-03 audit |
| Gaussian centers | `lmain.xyz` | Yes | `GaussianField.means3d` | Plans 03 and 06 |
| Rotation | `lmain.rot_maps` | Yes | `GaussianField.rotations` | Plans 03 and 07 |
| Scale | `lmain.scale_maps` | Yes | `GaussianField.scales` | Plans 03 and 07 |
| Opacity | `lmain.opacity_maps` | Yes | `GaussianField.opacities` | Plans 03 and 07 |
| Renderer | `pts2render` -> `gaussian_renderer.render` | Yes, CUDA extension required | `RenderOutput.image` | Plans 03 and 07 |
| Left camera | SCARED `lmain` intr/extr/raster fields | Yes | Future verified `CameraBatch` bridge | Plan 02 |
| Right camera | SCARED `rmain.img`, `rmain.intr` only | Pose not exposed | Cannot fully map yet | Plan 02 blocker |

This inventory authorizes observation only. It does not authorize uncertainty,
geometry reimplementation, covariance, cross-view loss, or any Phase II code.

## Parity state

CPU tests compare direct upstream-shaped tensors with translated disparity,
Gaussian attributes, renderer RGB, layout changes, and the exact RGB range
conversion. They freeze zero tolerance for fields that should be identical.
The `TensorParity` harness records shape, dtype, device, max/mean absolute
difference, and tolerances.

Official GPU inference and end-to-end native-versus-adapter parity were not run:
there is no verified checkpoint or legal fixture, the compiled runtime is not
available, and key camera/input conventions remain blocked on Plan 02. Those
acceptance criteria remain open; synthetic conversion parity is not presented
as baseline reproduction.

## Server runtime update (2026-08-19)

The server has an RTX 2050 and a CUDA-capable PyTorch 2.9.0+cu128 runtime, but
not the upstream's Python 3.10/PyTorch 2.0.1/CUDA 11.8 stack. It has neither a
CUDA toolkit (`nvcc`/`CUDA_HOME`) nor MSVC compiler tools. The official Graphdeco
rasterizer source was identified at
`59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d`; its direct build attempt correctly
stopped before compilation because `CUDA_HOME` is absent. The unmodified
upstream model also stops before rasterization because its configured
`reg_cuda` correlation implementation requires an absent `corr_sampler`.

The complete compatibility matrix, external-source provenance, license states,
synthetic diagnostic, and blocked numerical-parity table are in
`docs/endo_e2e_gs_runtime_parity_record.md`.
