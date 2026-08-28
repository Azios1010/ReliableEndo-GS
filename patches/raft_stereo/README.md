# RAFT-Stereo `corr_sampler` Compatibility Patch

## Metadata

| Field | Value |
| --- | --- |
| Classification | `BUILD_API_COMPATIBILITY_ONLY` |
| Target repository | `https://github.com/princeton-vl/RAFT-Stereo` |
| Target commit SHA | `6068c1a26f84f8132de10f60b2bc0ce61568e085` |
| Target file | `sampler/sampler_kernel.cu` |
| Target license | MIT (declared in `princeton-vl/RAFT-Stereo`) |
| Consumed by | `third_party/endo_e2e_gs` (`core/corr.py` default `corr_implementation=reg_cuda`) |
| Scientific status | **Unchanged science**: API dispatch fix only; no mathematical or algorithmic alteration |
| Runtime equivalence requirement | Verification against baseline mathematical reference required prior to claiming GPU parity |

## Purpose and Rationale

The pinned Endo-E2E-GS baseline uses `corr_implementation=reg_cuda` by default in `core/corr.py`, which imports the external compiled extension `corr_sampler`.
Upstream `corr_sampler` source from `princeton-vl/RAFT-Stereo` at commit `6068c1a26f84f8132de10f60b2bc0ce61568e085` was written for legacy PyTorch C++ APIs where `AT_DISPATCH_FLOATING_TYPES_AND_HALF` took `volume.type()`.

In PyTorch 2.x+, `Tensor.type()` is deprecated and cannot be used as the dispatch operand in `AT_DISPATCH_*` macros. Passing `volume.type()` results in compilation errors. Replacing `volume.type()` with `volume.scalar_type()` conforms to the current ATen scalar-type dispatch interface (`at::ScalarType`).

## Interface and Scope

This patch modifies **only** the two dispatch macro calls in `sampler/sampler_kernel.cu`:
1. `sampler_cuda_forward`: `volume.type()` -> `volume.scalar_type()`
2. `sampler_cuda_backward`: `volume.type()` -> `volume.scalar_type()`

It preserves exact C++ / Python binding interfaces in `sampler/sampler.cpp`:
- `forward(volume, coords, radius)`
- `backward(volume, coords, grad_output, radius)`

It does not change kernel dimensions, block/grid launch parameters, bounds checking, index calculation, floating-point math, or output memory layouts.

## Why an Adapter-Only Solution is Insufficient

Endo-E2E-GS's `StereoEndoModel` initializes `CorrBlockFast1D` with `reg_cuda`, which directly invokes `CorrSampler.apply` calling `corr_sampler.forward`. Falling back silently to `reg` (pure PyTorch) would mutate baseline execution semantics without approval and could mask parity discrepancies. Therefore, building `corr_sampler` with this isolated API compatibility patch is required for genuine baseline GPU reproduction.

## Application and Verification Instructions

1. Clone the exact pinned RAFT-Stereo commit into a temporary build location:
   ```bash
   git clone https://github.com/princeton-vl/RAFT-Stereo.git /tmp/raft_stereo
   cd /tmp/raft_stereo
   git checkout 6068c1a26f84f8132de10f60b2bc0ce61568e085
   ```
2. Apply the patch:
   ```bash
   git apply /path/to/ReliableEndo-GS/patches/raft_stereo/0001-corr-sampler-scalar-type-compatibility.patch
   ```
3. Build the extension in a CUDA-enabled environment (requires MSVC/GCC + CUDA toolkit + torch):
   ```bash
   cd sampler
   python setup.py install
   ```
4. Verify import and function signatures:
   ```python
   import corr_sampler

   assert hasattr(corr_sampler, "forward")
   assert hasattr(corr_sampler, "backward")
   ```
