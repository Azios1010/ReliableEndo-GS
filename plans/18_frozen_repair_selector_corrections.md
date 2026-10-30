# Frozen repair / selector protocol corrections

Authorized by the user on 2026-10-28 following the implementation audit and
research recommendations. This is a development protocol, not Gate I/II evidence.
No new student, repair or selector experiment is authorized to bypass its gates.
Baseline continuation uses its existing external source snapshot and recipe.

## Fixed design before new outcome collection

- Keep the selection-free KD student frozen. Keep the existing Stage A repair
  training objective; discard its temporary selector. Require an independently
  evaluated, hashed Stage A artifact before Stage B fitting.
- Use only train_1236 for FIT/VALIDATION/CALIBRATION, split by whole keyframes
  within the manifest's four datasets (7/4/4 keyframes). Partition seed 0 is
  independent of selector fit seeds 0, 1, 2. Development is val_7 only. Datasets
  4/5 and final_test 8/9 are excluded. Never move development into fitting after
  seeing a distribution gap.
- Primary automatic region protocol: deterministic spatially spread support-grid
  patches intersected with deployment sensor/prediction support. It is a bounded
  candidate sampler, not tissue segmentation. No manual ROI, GT, learned error
  score or additional backbone determines the patches. All arms share them.
  Stage A can use the same producer. Legacy grid artifacts need explicit read-only
  import and a fresh Gate A evaluation under the collection candidate protocol.
- Collect one superset feature bank (61 statistics plus three width-C student
  token pools), raw 12-component physical changes, measured action costs and
  unknown/unavailable outcomes before any fitting. Infer C from the checkpoint.
- Compare C0 statistics/union risk, C1 tokens/local-global risk and C2 signed
  gain/local-global risk, each with the same partition and three fit seeds.
  C1 is a candidate, not a proven winner. Keep legacy geometry utility as a
  control; the render utility arm predicts mean left/right PSNR gain in dB.
  Depth, SSIM, collateral RGB and coverage remain physical guards. Cost is
  subtracted exactly once by the existing decision policy, never in labels.
- Gate A follows Plan 16: positive paired keyframe-bootstrap gain, consistent
  sign in at least three of four development keyframes, violations <=25%.
  Gate B requires positive oracle headroom over the best predeclared fixed
  policy, a positive lower 95% keyframe-bootstrap bound, and unique positive
  contributions from each repair in at least two FIT keyframes. No arbitrary
  per-action 10% frequency quota. Gate B uses FIT outcomes only; calibration
  and development cannot decide whether to train. Failure is preserved.
- Evaluate STOP, cyclic-region fixed stereo, cyclic-region fixed Gaussian and
  one-action-per-frame oracle under the same declared repair budget. Report
  both-view quality, unknown coverage, per-event/action calibration and
  keyframe intervals. Three selector seeds are not three independent students.
- Select the variant on VALIDATION and operating cutoffs on CALIBRATION only.
  Four predeclared harm budgets: 1%, 5%, 10%, 20%, each capped by the best fixed
  policy's CAL harm rate. Unsupported class/keyframe support forces abstention.
  Empirical CAL compliance is not a finite-sample safety guarantee.
- Report LPIPS only with pinned pretrained/calibrated local weights. Missing
  weights produce an explicit unavailable status or a required-metric failure.
  Latency/FPS measurement remains deferred by the user's earlier instruction;
  training time and summed action timings do not establish end-to-end FPS.

## Evidence and promotion limits

All experiment launches must seal resolved configuration, source, frozen actor,
candidate protocol, worklist, partition and cost identities before outcomes.
Technical tests establish implementation behavior, not repair utility or paper
superiority. Keep negative gates and earlier failed selector evidence.
Non-inferiority margins and final_test remain outside this change.

The implementation report is `docs/repair_selector_revision_report_2026-10-28.md`.
