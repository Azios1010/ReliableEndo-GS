# Endo-E2E-GS runtime and parity record

## Record scope

This is a Plan 01 integration-runtime record, not a baseline reproduction
artifact. It records server-side capability observations and failed execution
boundaries for the exact unpatched Endo-E2E-GS source. It contains no dataset,
checkpoint, generated render, or scientific metric.

| Field | Value |
| --- | --- |
| ReliableEndo-GS commit | `988fee4` |
| Upstream repository | `Intelligent-Imaging-Center/Endo-E2E-GS` |
| Upstream commit | `186fa2b4a2159b28393492f6df1aa444b54391a8` |
| Upstream state | Clean and unpatched |
| Python | 3.14.3 |
| PyTorch | `2.9.0+cu128` |
| PyTorch CUDA build | 12.8 |
| GPU | NVIDIA GeForce RTX 2050, 4 GiB |
| Driver-reported CUDA | 13.1 |
| Checkpoint | None available or authorized; no SHA-256 to record |
| Fixture | Ephemeral synthetic zero stereo input, `[1,3,256,256]`, float32 CUDA; technical execution only |
| Native inference | Not completed; stopped before prediction at missing `corr_sampler` |
| Adapter inference | Not run because native upstream produced no output to translate |
| Native-versus-adapter parity | Not run; no comparable native output |

## Compatibility matrix

| Requirement | Pinned upstream expectation | Server observation | Status |
| --- | --- | --- | --- |
| Python | 3.10.13 | 3.14.3 | **UNCLEAR**: not an upstream-tested interpreter. |
| PyTorch | 2.0.1 | 2.9.0+cu128 | **UNCLEAR**: upstream environment is not reproduced. |
| CUDA toolkit | PyTorch CUDA 11.8 build environment | No toolkit, `nvcc`, or `CUDA_HOME`; driver advertises CUDA 13.1 | **BLOCKED** for extension compilation. |
| C++ compiler | Required by `torch.utils.cpp_extension.CUDAExtension` | No MSVC `cl.exe` visible | **BLOCKED** for extension compilation. |
| `diff_gaussian_rasterization` | External Graphdeco CUDA extension | Not installed; direct build attempt fails because `CUDA_HOME` is absent | **BLOCKED**. |
| GLM | Rasterizer Git submodule | Required at `5c46b9c07008ae65cb81ab79cd677ecc1934b903` by the selected rasterizer source | **NOT INITIALIZED**; irrelevant until a compatible toolchain exists. |
| RAFT `corr_sampler` | Required by Endo-E2E-GS default `reg_cuda` correlation mode | Not installed; the Endo-E2E-GS tree has no sampler source/build instructions | **BLOCKED**. |
| Other Python imports | `cv2`, SciPy, torchvision, YACS | Imports available | **VERIFIED** for model import only. |
| Checkpoint | Official/author-provided, identified hash | No authorized checkpoint present | **BLOCKED**. |

No package was upgraded, downgraded, or installed into the project runtime. The
attempted rasterizer editable install did not pass metadata generation and did
not create an importable package or compiled binary.

## External compiled-source provenance

| Component | Source | Immutable commit | License status | Build result |
| --- | --- | --- | --- | --- |
| Differential Gaussian Rasterization | `graphdeco-inria/diff-gaussian-rasterization` | `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d` | **VERIFIED**: `LICENSE.md` permits research/non-commercial use and imposes redistribution/notice conditions. Compatibility with the upstream root MIT notice remains **UNCLEAR**. | `pip install --no-deps --no-build-isolation -e <source>` stopped at `CUDA_HOME` absent. |
| GLM dependency | `g-truc/glm` submodule of rasterizer | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` | **NOT REVIEWED** separately; no build occurred. | Not initialized. |
| RAFT correlation sampler | Official RAFT-Stereo documents `cd sampler && python setup.py install` | Current RAFT-Stereo `main` observed at `6e93ed2169bd858dbb43033988563f3b0bb49506`; exact source revision used by Endo-E2E-GS is not attributed | **UNCLEAR** for copied-source provenance; RAFT-Stereo repository itself declares MIT. | Not attempted: source provenance is unpinned and the required compiler/toolkit are absent. |

The Endo-E2E-GS README names Graphdeco's `gaussian-splatting` repository but
does not pin it. The direct rasterizer repository is the module actually
imported by `gaussian_renderer/__init__.py`; its selected commit is recorded
here only for reproducible future setup, not as a completed dependency.

## Native execution attempts

1. Importing `core.corr` and `lib.network` from the clean pinned upstream tree
   succeeded on the server. Importing `gaussian_renderer` correctly failed
   through ReliableEndo-GS capability detection because
   `diff_gaussian_rasterization` is absent.
2. An unmodified `StereoEndoModel(..., with_gs_render=True)` was invoked in
   evaluation mode using the upstream stage-2 configuration and a synthetic
   256-pixel-square CUDA input. No checkpoint was loaded. The first meaningful
   runtime failure was `NameError: name 'corr_sampler' is not defined` inside
   `core/corr.py`, because the upstream default is `corr_implementation=reg_cuda`.
3. A 64-pixel-square synthetic diagnostic failed earlier because the correlation
   pyramid became too small. It is retained only as an input-size diagnostic,
   not as an inference result.

The synthetic fixture was not saved. It is not a legal dataset fixture, not a
baseline output, and not evidence of scientific model quality.

## Checkpoint and licensing status

| Component | Status | Evidence and consequence |
| --- | --- | --- |
| Endo-E2E-GS root source | **VERIFIED** | Root `LICENSE` is MIT and preserved in the pinned submodule. |
| Embedded Graphdeco-derived files | **UNCLEAR** | Their headers point to Graphdeco `LICENSE.md`; the direct rasterizer source supplies that research-only license, but the Endo-E2E-GS tree omits it and attribution/license reconciliation has not been provided. |
| Rasterizer redistribution | **UNCLEAR** | Research use terms are observable; redistribution must retain notices/license and requires project-level review. No source copy or binary is committed. |
| Checkpoint/weights | **BLOCKED** | The official README supplies only a filename convention. No official/author-provided download, hash, or explicit terms were available. |
| Dataset fixture | **BLOCKED** | No authorized official input was supplied; Plan 02 owns real SCARED inspection. |

No legal conclusion is made beyond the text available in the identified source
files. A checkpoint-dependent parity claim remains prohibited.

## Numerical parity table

| Quantity | Native output | Adapter output | Shape/dtype/device | Error result | Status |
| --- | --- | --- | --- | --- | --- |
| Disparity | Unavailable | Unavailable | N/A | N/A | BLOCKED by `corr_sampler` and checkpoint. |
| Gaussian means | Unavailable | Unavailable | N/A | N/A | BLOCKED. |
| RGB/colors | Unavailable | Unavailable | N/A | N/A | BLOCKED. |
| Rotation | Unavailable | Unavailable | N/A | N/A | BLOCKED. |
| Scale | Unavailable | Unavailable | N/A | N/A | BLOCKED. |
| Opacity | Unavailable | Unavailable | N/A | N/A | BLOCKED. |
| Rendered RGB | Unavailable | Unavailable | N/A | N/A | BLOCKED by rasterizer. |

The existing CPU contract fixtures still show exact translation for mapped
pass-through fields and the documented RGB range conversion. They are not
substituted for this table.

## Required next action to complete Plan 01

Provision one dedicated, reproducible upstream runtime with Python 3.10,
PyTorch 2.0.1/CUDA 11.8, matching CUDA toolkit and MSVC toolchain; pin and build
both `diff_gaussian_rasterization` and the exact compatible RAFT sampler source.
Obtain an authorized official checkpoint and legally usable fixture with hash
and terms. Then run the unmodified upstream path first and compare its produced
tensors with adapter translations on the identical input. Plan 02 must not
start until that evidence is recorded or an approved Plan 01 pivot is made.

## Runtime and compatibility update (2026-08-27)

| Field | Value | Status |
| --- | --- | --- |
| GPU hardware | NVIDIA GeForce RTX 5070 Ti x2 | Prepared |
| Python | 3.10.11 | Prepared |
| PyTorch | `2.7.1+cu128` | Prepared |
| CUDA toolkit | Private CUDA 12.8.1 toolkit | Prepared |
| `diff_gaussian_rasterization` | Built and smoke-checked in earlier environment session | Not rerun in this CPU/static scope |
| `corr_sampler` provenance | `https://github.com/princeton-vl/RAFT-Stereo` @ `6068c1a26f84f8132de10f60b2bc0ce61568e085` | **RESOLVED** |
| `corr_sampler` compatibility | Patch prepared under `patches/raft_stereo/0001-corr-sampler-scalar-type-compatibility.patch` (`BUILD_API_COMPATIBILITY_ONLY`, `volume.scalar_type()` dispatch) | **PREPARED / STATICALLY VALIDATED**; not GPU-tested |
| Checkpoint | Official/author-provided checkpoint authorization and download | **BLOCKED** |
| Dataset fixture | Authorized legal SCARED fixture | **BLOCKED** on Plan 02 data inspection |
| Native inference & parity | End-to-end native execution vs adapter | **NOT RUN** (hard no-GPU rule in effect) |
| Milestone status | Plan 01 | **IN PROGRESS** (Plan 02 early implementation in progress) |

## `corr_sampler` focused GPU build and smoke attempt (2026-08-28)

The pinned external RAFT-Stereo source at
`6068c1a26f84f8132de10f60b2bc0ce61568e085` was freshly cloned outside this
repository and the patch at
`patches/raft_stereo/0001-corr-sampler-scalar-type-compatibility.patch` applied
cleanly. Its external source diff was limited to the two intended
`volume.type()` -> `volume.scalar_type()` dispatch operands in
`sampler/sampler_kernel.cu`; neither `third_party/endo_e2e_gs` nor any baseline
source was modified.

Before GPU execution, `CUDA_VISIBLE_DEVICES=1` produced exactly one logical
PyTorch `cuda:0`; its properties were `NVIDIA GeForce RTX 5070 Ti`, capability
`(12, 0)`, matching physical GPU 1 reported by `nvidia-smi`. The requested
repository-local `.venv` and private CUDA 12.8.1 toolkit were
not present in the execution environment, so a coordinator-authorized bounded
attempt used the repository-local `.venv` (Python 3.10.11, PyTorch
`2.7.1+cu128`) with the only installed compiler, CUDA 12.6.85.

**Result: FAIL / BLOCKED.** With MSVC initialized and
`TORCH_CUDA_ARCH_LIST=12.0`, the external build reached `nvcc` but CUDA 12.6
rejected `sm_120` (`nvcc fatal: Value 'sm_120' is not defined for option
'gpu-name'`); `nvcc --list-gpu-arch` only lists through `compute_90`. No
extension artifact was produced, `import corr_sampler` failed with
`ModuleNotFoundError`, and the required forward/backward presence check, tiny
GPU forward/backward, finite-value/gradient, shape, and CUDA-error checks were
therefore not run. Provision the actual CUDA 12.8.1 toolkit (with `sm_120`
support) before repeating the same external build and focused smoke; do not use
the pure-PyTorch `reg` fallback.

## `corr_sampler` successful isolated GPU build and smoke retry (2026-08-28)

The later retry used the external RAFT-Stereo checkout pinned at
`6068c1a26f84f8132de10f60b2bc0ce61568e085`. The exact compatibility patch was
limited to the two `volume.type()` -> `volume.scalar_type()` dispatch operands
in `sampler/sampler_kernel.cu`; the external build used the private CUDA 12.8.1
toolkit, whose `nvcc --version` reported `V12.8.93`, and compiled for `sm_120`.

With physical GPU 1 masked as the only visible device, PyTorch exposed it as
logical `cuda:0`: an NVIDIA GeForce RTX 5070 Ti with capability `(12, 0)`. The
focused smoke successfully imported `corr_sampler`, found both `forward` and
`backward`, and ran the tiny GPU forward/backward checks with expected shapes:
`volume` `(1, 2, 3, 4)`, `coords` `(1, 2, 2, 3)`, forward `corr` `(1, 3, 2, 3)`,
and backward `volume_grad` `(1, 2, 3, 4)`. Output and gradient values were
finite, the gradient was nonzero, and synchronization before and after the
forward/backward calls completed without CUDA errors.

**Result: PASS for isolated `corr_sampler` build and focused GPU smoke only.**
No `third_party/endo_e2e_gs` or Plan 02 changes were made. Native Endo-E2E-GS
inference and native-versus-adapter parity remain blocked on an authorized
checkpoint and a legally usable fixture; this record does not claim either.
