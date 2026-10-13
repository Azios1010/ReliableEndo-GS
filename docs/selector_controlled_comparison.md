# Controlled selector comparison and checkpoint selection

## Scope

User-approved development diagnostic,2026-10-12. Original SCARED datasets1–7;
reserved datasets8–9 untouched. Student/repairs are fixed only for this comparison.
No teacher inference, dependency installation or large joint training.

External evidence run: `selector-controlled-rzzeletm` under the external run root.
Raw outcomes, checkpoints, source snapshots and reports remain outside the repository.

## Controlled evidence

Collected584 frames:330 optimizer FIT,86 internal stopping on3 withheld FIT
keyframes and168 original CAL. Evaluated140 original VAL frames,20 per dataset.
Original role manifests stay unchanged; worklists were sealed before GT.
The historical student initialization may have seen this parent cohort.

Compared shared-action/shared optimization, separate-action/shared optimization,
and separate-action/action-local optimization under the same frozen model/action
state, features, normalization, utility and physical/risk policy, across3 seeds.
Fixed256-step controls remove differing stopping rules: every arm and seed selects
STOP,stereo and Gaussian. Separated heads do not prevent stereo learning.
The historical head also selects all3 on current logs, but its utility, fitting
exposure and calibration differ, so its transfer is not an architecture ablation.

## Checkpoint-selection remediation

The existing per-component guard retains step25 across3 seeds. At step50 both
action objectives, both risk BCEs and ranking improve; it rejects solely because
stereo global regression increases slightly. The retained raw heads largely STOP.

`SelectorRefitConfig(validation_guard_mode="action_objectives")` is a new opt-in:
select one whole selector snapshot using the unchanged normalized total objective,
with accepted-minimum guards on each action's weighted global/local/risk objective,
each action's risk BCE independently,and ranking. Default `component_minima`
behavior and historical config hashes remain compatible. Shared balancing rejects
the new mode. Utility labels, physical guards, risk cutoffs and action quotas do
not change. All3 remediation seeds select gain/risk from step50.

## Native seed0 confirmation on140 identical frames

| Policy | STOP / stereo / Gaussian | Mean utility | Guard violations | Unknown |
| --- | --- | --- | --- | --- |
| Original128-step joint selector |12 /0 /128 |0.131801 |25 |4 |
| Remediated stationary refit + CAL |31 /43 /66 |0.129088 |16 |4 |
| Offline physical oracle |21 /25 /94 |0.173621 |0 |0 |

Original/refit means use the same136 known frames. Seed0 was predeclared;
there is no selection of seed by VAL performance. Native predictions match replay;
base render metrics match across runs and scientific weights remain unchanged.
This is post hoc development remediation,not an untouched final test.

Stereo risk Brier improves0.24666→0.15985; Gaussian0.18569→0.17919.
True-safe gain ranking improves75.64%→84.62%; correct stereo-best gain rankings
0/36→20/36. This supports useful learning signal,not complete risk discrimination.

Tradeoffs persist: utility decreases slightly,16 guard violations remain,
full-image depth MAE changes only~0.0005mm and render PSNR changes~0.01dB.
Estimated mean action cost rises98.28→104.60ms. No superiority or FPS claim.
Remediation CAL utility spans0.1141–0.1291 with16–22 violations across3 seeds.
Rendered-depth metrics and paper-protocol parity remain unavailable.

## Verification and limits

9 focused guard/config-hash tests passed; Ruff,mypy2 source files and whitespace
checks passed. Full CI not run. Every calibrated stationary artifact round-tripped
against its scientific bindings; the input checkpoint bytes are unchanged.

Cold CPU balanced-backward diagnostics aborted with native double-free; the GPU
control and CPU refit after verified CUDA initialization completed. Runtime logs
preserve failed attempts; the allocator cause is not established. No packages were
installed. Source before/after snapshots and complete patch are archived externally.
Native confirmation explicitly pins the historical checkpoint SHA with read-only
compatibility review and binds the new head to the new fitting-source identity.

Refit calibration is valid only for this frozen student/repair artifact. Joint
updates require fresh outcomes and eventual calibration; the old calibrated head
must not be attached to changed repairs. A larger joint run is a separate next
experiment. This diagnostic does not pass a research gate or establish generalization.
