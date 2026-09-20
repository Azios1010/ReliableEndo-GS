# P1b stereo-consensus observer feasibility design

Status: **DESIGN ONLY / NOT IMPLEMENTED**

This note is part of P1b — Reliability Signal Rescue.  It does not authorize
training, a generator implementation, checkpoint creation, renderer changes,
or access to the untouched final sequences.

## Proposed later study

The proposed observer would take the rectified pair as a joint input:

```text
L ───┐
     ├──> shared stereo representation Z ───> L_hat
R ───┘                                  └──> R_hat
```

The reliability evidence would be the two reconstruction residuals

```text
E_L = distance(L, L_hat)
E_R = distance(R, R_hat)
```

and their support masks.  This is a scene-consensus signal, not the existing
one-directional photometric warp residual U4.

## Scientific justification

**MAYBE — justified as a later P1c feasibility study, not as a P1b result.**

The observer is scientifically distinct from U4 because it can test whether a
shared latent scene explains both views at once.  That distinction is useful
if the independent P1b dynamics, geometry, and observation channels survive
without fusion.  It is not justified as an immediate implementation because a
learned observer could absorb appearance, calibration, and reconstruction
shortcuts, making it difficult to attribute residuals to geometric reliability.

## Required preregistration before implementation

1. Freeze the input normalization, image resolution, augmentations, and stereo
   pair ordering.
2. Freeze the latent bottleneck and decoder capacity independently of the
   reliability evaluation set.
3. Define whether the observer is trained on fit sequences only and whether
   holdout images are ever used for early stopping or architecture selection.
4. Define fixed image-space, gradient, and structural losses, plus explicit
   invalid/occluded support handling.
5. Compare against a non-learned U4/U4b control and a view-independent image
   reconstruction control.
6. Require both-view residual reporting, support coverage, cross-sequence
   bootstrap, and extra forward/memory cost before making any reliability
   claim.

## Recommended destination

`P1c` is the safer destination because the observer would introduce a new
learned signal and requires its own feasibility gate.  It should move to
Phase II only if it is frozen as an auxiliary diagnostic and does not alter
the Phase-I representation or provide an unregistered routing label.

No generator, training loop, or learned fusion is implemented in P1b.
