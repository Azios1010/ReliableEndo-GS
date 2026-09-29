# A03 geometry-derived Gaussian priors

Status: IMPLEMENTED engineering package; scientific acceptance remains PROPOSED.

The CPU constructor in `reliable_endo_gs.gaussians.priors` reuses
`reconstruct_rectified` for metric left-camera centres and
`build_surface_covariance` for the canonical support covariance.  Its local
surface estimator masks invalid neighbours and optical-axis discontinuities,
orients normals toward the camera, builds the deterministic camera-relative
`R0=[t1,t2,n]` gauge with a least-aligned-axis branch, and marks conservative
bounded isotropic fallbacks.  `S0`, normalized wxyz rotations, `alpha0`, RGB,
validity and source pixel correspondence are returned together.

`gaussians.sampling` provides deterministic dense, fixed-grid and sparse
row-major policies with explicit per-image caps.  Dense mode enumerates every
valid pixel; fixed-grid and sparse modes use their configured stride.  Discrete
masks, branches and indices are not differentiable; selected metric point,
normal and scale arithmetic retains PyTorch gradients.  Padded slots are
finite placeholders with `valid_mask=False` and must be omitted by a renderer.

Focused CPU coverage is in
`tests/architecture/test_a03_geometry_gaussian_priors.py`, including planes,
frames, covariance/quaternion identities, invalid and discontinuous support,
curved surfaces, metric footprint scaling, source-feature correspondence,
sampling correspondence/caps, gradients and forbidden imports.  The focused
A03 plus canonical geometry suite passed on the available CPU interpreter.

No teacher winner, native CUDA/rasterizer parity, runtime measurement, quality
gain, opacity interpretation or experiment result is claimed.  A04 remains
out of scope; future student, residual-head, repair, oracle, selector and RL
work must use separately authorized tasks.
