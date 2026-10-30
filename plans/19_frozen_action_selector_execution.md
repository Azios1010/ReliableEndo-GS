# Authorized frozen action/selector execution, 2026-10-28

The user explicitly requested investigating KD's latency, fixing it where needed,
training actions and selectors through the previously decided phases, and ending
the Codex work session once the detached job is stable with a detailed report.
This authorizes unattended chaining of the existing development protocol. It
supersedes Plan 16/17's historical requirement for a separate launch instruction
between stages; it does not waive any scientific gate or authorize final_test.

## Fixed sequence and experiment

1. Profile frozen final baseline and final KD on physical GPU0. Preserve their
   completed outputs. Separate device component times from host synchronization
   waits. Optimize execution only after measurement and compare against the native
   FP32 computation on the existing 56-frame validation worklist.
2. A0: exact checkpoint/architecture loading; bounded three-update action pilot;
   unchanged student/head/model-Adam; both repairs update; finite native losses.
3. Stage A: `student_frozen_repair.yaml`, **one joint epoch / 13,150 samples** of
   unchanged train_1236, seed 20261006, final KD model-only initialization. Frozen
   student and attribute head, existing repair objectives, Plan 18 automatic
   support-grid candidates. The temporary selector is discarded.
4. Verify and hash final Stage A actor. Prepare GT-free TRAIN cost calibration.
   Evaluate STOP, fixed stereo and fixed Gaussian on the presealed 64-frame val_7
   cohort with pinned LPIPS; derive Gate A with the unchanged Plan 18 rules.
5. Gate A pass only: collect one stationary feature/outcome bank on 240 presealed
   TRAIN frames, 16 per keyframe. Partition seed 0 gives 7 FIT / 4 VALIDATION /
   4 CALIBRATION keyframes. Report unknown and unavailable outcomes.
6. FIT-only Gate B pass for a utility arm only: C0/C1/C2, fit seeds 0/1/2,
   geometry and render-PSNR utility arms, four predeclared CAL harm budgets.
   Keep max_steps 1000, min_steps 50, validation every 25, patience 6. These are
   optimization stopping limits, not a promise of 1000 steps for every fit.
7. Variant choice remains VALIDATION-only, cutoffs CAL-only. Native-evaluate all
   chosen artifacts/seeds on the same 64 development IDs, then apply the existing
   promotion rule. Failure stops the affected stage and retains negative evidence.

No thresholds, sample selection, epochs, utility weights or budgets are changed in
response to new repair/selector results. A failed Gate A or all failed Gate B arms
means no selector fit. This is an explicitly authorized development branch; it
does not certify the program's Phase I artifact or bypass Gate I for a paper claim.

## Execution and evidence

Use an isolated source/config snapshot, new external outputs and physical GPU0;
leave the unrelated GPU1 job running. Stage bytes to Linux before training without
changing samples, decoded tensors or labels. Seal source/config/checkpoint/worklist,
partition, hardware and workspace Git provenance. Preserve completed baseline and KD.

A detached tmux supervisor owns Stage A and subsequent gated commands. It uses
strict checkpoint resume with bounded explicit OOM/allocation/I/O retries, disk
reserve and checkpoint pruning. Once at least 100 updates and a checkpoint exist,
verify recent finite losses and exact frozen model/Adam equality before handing off.
Record ongoing status, logs, gate reports and final artifact paths. Closing Codex
must not send signals to this supervisor.

The user now authorizes latency investigation, superseding the earlier FPS
deferral for this task. Initial timing uses preloaded inputs, batch1 and both
renders; its exclusions must remain explicit. No matched-cost final evaluation,
paper superiority or high-FPS claim follows solely from this optimization.
