# Graphdeco Rasterizer CUDA 12.8 Compatibility Patch

## Metadata

| Field | Value |
| --- | --- |
| Classification | `BUILD_API_COMPATIBILITY_ONLY` |
| Target repository | `https://github.com/graphdeco-inria/diff-gaussian-rasterization` |
| Target commit | `59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d` |
| Target file | `cuda_rasterizer/rasterizer_impl.h` |
| Consumed by | `third_party/endo_e2e_gs/gaussian_renderer` |
| Scientific status | Unchanged science: standard-library declarations only |

## Purpose

The pinned source uses `std::uintptr_t`, `uint32_t`, and `uint64_t` without
directly including `<cstddef>` and `<cstdint>`. CUDA 12.8 compilation therefore
fails before any kernel is built. This patch adds only those declarations; it
does not change the Python API, kernels, launch dimensions, indexing, or
floating-point operations.

## Application

```bash
git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git /tmp/diff_gaussian_rasterization
cd /tmp/diff_gaussian_rasterization
git checkout 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
git submodule update --init --recursive
git apply /path/to/ReliableEndo-GS/patches/diff_gaussian_rasterization/0001-add-standard-integer-headers.patch
```

Build only with the intended Python/PyTorch runtime and a CUDA toolchain that
supports the target GPU. Verify that `GaussianRasterizer(...)` returns exactly
`(rendered_image, radii)`, then run the project Stage-2 smoke before any
official artifact.
