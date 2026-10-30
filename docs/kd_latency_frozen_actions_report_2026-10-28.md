# KD latency investigation and frozen action/selector execution — 2026-10-28

## Tóm tắt để kiểm tra

KD chậm chủ yếu ở **GroupNorm/decoder toàn độ phân giải**, không phải do I/O.
Đã profile thật trên GPU0: stereo của KD khoảng49,20 ms, riêng decoder31,20 ms;
geometry prior14,43 ms; mỗi view render4,43 ms. Baseline dùng AMP cho stereo,
KD giữ FP32 như checkpoint. Các thành phần khác nhau nên không suy tốc độ từ
số parameter hoặc số Gaussian.

Đã thử tối ưu lớn nhưng **không đưa vào run**: GroupNorm song song đạt khoảng44 ms
cho một frame, nhưng sai lệch disparity tối đa0,02445 px vượt ngưỡng0,001 px đặt
trước; CUDA kernel giữ thứ tự Welford đạt bitwise trên ba shape nhưng gần như
không nhanh hơn. Giữ GroupNorm native, precision, kiến trúc và trọng số gốc.
Chỉ sửa việc tạo lặp index của matching:64 lần xuống1 lần, không đổi phép tính.
Phép sửa nhỏ này **chưa cho thấy tăng tốc median đáng kể**:75,77→75,96 ms trong
lượt kiểm tra cặp, vì vậy không dùng để tuyên bố tăng FPS.

Đã kiểm tra **56 frame thật**: disparity, correspondence, validity và ảnh cả hai
view khớp; sai lệch ảnh tối đa0 và disparity tối đa0. PSNR/SSIM/LPIPS/EPE/depth MAE
có sai lệch0 so với cả control native lẫn benchmark KD đã lưu. Bộ CPU đạt491 pass,
13 skip CUDA. Pilot native3 update đạt: student/head và model Adam giữ nguyên
bitwise; stereo repair và Gaussian repair đều cập nhật3 lần; model update0 lần.

Run mới theo đúng thứ tự: **Stage A1 epoch/13.150 mẫu → kiểm tra/freeze → Gate A →
collect240 TRAIN frame → Gate B → C0/C1/C2 ×3 seed × utility arm đạt gate → CAL và
native development/promotion**. Không lấy selector tạm của Stage A làm kết quả
Stage B. Gate không đạt thì ghi NO-GO và dừng phần phụ thuộc, không sửa threshold.

Cohort đã niêm phong trước outcome repair:240 TRAIN/64 development, chia keyframe
7 FIT/4 VAL/4 CAL. Dùng patch automatic từ support, không dùng manual ROI hay GT
để chọn vùng. Toàn bộ13.150 mẫu train đã có byte-cache Linux, không đổi split,
không bỏ mẫu. KD, baseline và các kết quả cũ được giữ nguyên.

Đây vẫn là nghiên cứu phát triển có gate. Action/selector có thể phục hồi chất
lượng hoặc tiết kiệm phần repair, nhưng không xóa chi phí GroupNorm của base KD.
Chưa có cơ sở nói hệ thống nhanh hơn baseline. Thay kiến trúc/retrain student,
multi-seed student, non-inferiority margin và final_test8/9 nằm ngoài lượt này.
Trạng thái bàn giao thực tế và checkpoint phải đọc ở phần cuối, không coi việc
chuẩn bị cấu hình là đã chạy ổn định.

## Scope and current status

The user requested explaining KD's latency, fixing it where appropriate, running
the previously agreed action/selector phases, and ending the Codex work session
after the detached job is stable. The dated execution authority is
[Plan 19](../plans/19_frozen_action_selector_execution.md); scientific thresholds
and the experimental design remain those of Plans 16–18.

Execution root: `/home/bkai/.local/share/reliableendo-gs/runs/phase2-frozen-actions-20261028T114155Z`.
The completed baseline and KD checkpoints and their original comparison are
preserved. This run is a gated development experiment; it does not certify Gate I
or authorize datasets 8/9, new student training, or a paper claim.

**Status after native readiness:** CPU validation and56-frame native regression
passed; the three-update native pilot passed; full Stage A started on GPU0 under
detached tmux. A final handoff update below records the observed stable
training/checkpoint state. Completion of the full epoch and gated Stage B remains
pending at handoff.

## Why KD is slower

The prior completed comparison measured 75.25 ms median for KD and 31.95 ms for
the baseline. A new read-only diagnostic on the same GPU0, with the first eight
pre-existing val_7 IDs, batch1, preloaded inputs, both renders, ten warmups and
100 repetitions, measured 77.39 ms and 32.05 ms. These are separate measured runs;
the original generated benchmark was not edited.

CUDA-event component measurements from 20 instrumented forwards:

| Component | KD median device time | Baseline median device time |
| --- | ---: | ---: |
| Stereo prediction, including features and decoder | 49.20 ms | Encoder 3.29 ms + RAFT 5.68 ms |
| Matching including both feature branches | 17.94 ms | Included above |
| Full-resolution residual decoder | 31.20 ms | No equivalent KD decoder |
| Geometry prior | 14.43 ms | Different geometry construction |
| Surface frame, nested inside geometry prior | 6.79 ms | Not the same computation |
| Gaussian attribute head | 1.09 ms | Parameter head 8.80 ms |
| Gaussian field composition | 3.92 ms | Different construction |
| Single-view renderer, two calls per frame | 4.43 ms | 4.83 ms |

Nested component times must not be added together. CUDA events and CPU wall
measurements answer different questions: the camera validation appears to wait
about 32 ms in the host profile because earlier queued stereo kernels finish
there. Removing that validation would not remove the queued decoder work.

The main bottleneck is native single-group GroupNorm across the full image. The
operator trace attributes 25.89 ms of device work to GroupNorm; the network has
23 GroupNorm layers, with the feature branch used twice, giving 40 calls per
forward. The learned decoder operates at 512×640 in FP32. PyTorch 2.7.1's native
GroupNorm uses a block for each sample/group, with long per-thread Welford
sequences. The installed header specifies 512 reduction threads. This is a
large single-group reduction with limited parallelism, rather than an I/O stall.
The pinned implementation is documented in the
[PyTorch 2.7.1 CUDA source](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/group_norm_kernel.cu).

The comparison also has a declared precision difference: baseline stereo uses
its Stage2 AMP path; KD uses its checkpoint-declared FP32 path. Parameter count
alone does not predict latency. This run does not silently enable AMP for KD or
remove its decoder, regularizer, Gaussian geometry, physical checks or renders.

The baseline's scales are predominantly tiny after its disclosed numerical
recovery, whereas KD has nonzero metric scales. On the profiled sample the
baseline emitted 327,680 valid primitives and KD 81,920. Nevertheless, the
measured per-view render times were similar, so primitive count or the baseline
scale collapse alone does not explain this latency gap.

Artifacts: `profile-before/profile.json`, `*-cpu.txt`, `*-operators.txt`, and
`*-trace.json` under the execution root. Instrumented component timings are
diagnostics, not an independent full deployment FPS claim. GPU1 continues other
user work; the whole host is not dedicated.

## Changes retained and alternatives rejected

The retained production change is small: `StereoStudent._matching` creates its
immutable disparity-column index once per matching pass, instead of allocating
the same index for every shift. Candidate values, padding, score arithmetic,
masking, softmax, precision, weights, gradients and correspondence are preserved.
There is no cross-frame tensor cache or mutable scientific state.

The exact patch is `runtime-fix.patch`. Its delta from the pre-investigation
snapshot changes only `src/reliable_endo_gs/reconstruction/student.py`.

The larger optimization experiments were **not retained**:

- Hierarchical `var_mean` GroupNorm reduced one-frame latency from 76.25 to
  44.09 ms, but changed disparity by up to 0.02445 px, exceeding the predeclared
  0.001 px limit. Decoder-only, feature-only and fused-affine variants also failed
  that limit. A fast method is not accepted merely because its image difference
  looks small.
- A CUDA trial preserved the installed 512-thread Welford sequence/tree while
  distributing warp work. It was bitwise equal to native GroupNorm for three
  tested tensor shapes, including 1×32×512×640. The large-shape timings were
  5.56 vs 5.53 ms, an improvement too small to justify a version-specific native
  extension. The earlier 256-thread prototype was neither equivalent nor faster;
  the actual installed 512-thread policy was then checked explicitly.
- All experimental GroupNorm code was removed from the working implementation
  and final source snapshot. Native PyTorch GroupNorm remains in production.
  Trial code/results remain outside the repository as negative evidence in
  `rejected-prototypes/`, `norm-candidates.json`, and `exact-norm-trial*.json`.

The retained index change cannot eliminate the decoder's architectural cost. No
large speedup or faster-than-baseline result is claimed. A substantial reduction
would likely need a separately approved student architecture/retraining or a
different numerical contract, with new matched evaluation. Routing can save
repair work; it does not remove the base student's GroupNorm cost. The requested
repair study still measures useful quality/utility behavior under the frozen
student, without forcing an efficiency conclusion.

The final accepted native regression compared the archived matching function
against index reuse on all56 pre-existing val_7 frames: maximum disparity and
left/right image error0, exact source indices/validity, and maximum difference0
for both-view PSNR/SSIM/LPIPS, EPE and depth MAE. Native metrics also matched the
archived KD benchmark with difference0. Two200-repeat diagnostic timings on the
first8 preloaded frames measured median75.77 ms native vs75.96 ms index reuse;
means79.77 vs76.32 ms were affected by the latency tail. This establishes
numerical equivalence, not a reliable end-to-end speedup.

## Fixed action/selector phases

| Phase | Fixed execution | Acceptance or stopping rule |
| --- | --- | --- |
| A0 | Read-only runtime regression, exact KD initialization, three-update native repair pilot | Finite losses, student/head/model-Adam unchanged, both repair heads update |
| Stage A | One joint epoch, 13,150 train_1236 samples, seed 20261006, final KD model-only initialization | Frozen student/head; verify final model and Adam bitwise; hash both trained repairs |
| Gate A | STOP/fixed stereo/fixed Gaussian on 64 sealed val_7 frames, pinned LPIPS | At least one repair: positive lower 95% keyframe-bootstrap PSNR-gain bound; positive sign in ≥3/4 keyframes; violations ≤25%; no selected unknowns |
| Collection | One frozen superset bank on 240 TRAIN frames, 16/keyframe | Same actors, candidates, features, raw physical deltas, unknowns and measured action costs; immutable shards |
| Gate B | FIT-only oracle vs best fixed policy | Positive lower 95% keyframe-bootstrap headroom; each repair uniquely contributes in ≥2 FIT keyframes |
| Selector fits | C0/C1/C2 × fit seeds 0/1/2 × passing geometry/render-PSNR utility arms | max_steps 1000, min_steps 50, validation every25, patience6; no DEV-driven fitting |
| CAL/DEV | Four harm budgets 1/5/10/20%; choose variant on VALIDATION, cutoff on CAL, native DEV all chosen seeds | Unsupported support abstains; existing ≥2/3-seed promotion rule; no failed-gate retuning |

The temporary Stage A selector is trained as required by the existing objective
and discarded. It is not the stationary Stage B selector. Gate A failure or
failure of every Gate B utility arm stops before final selector fitting and
retains a NO-GO result. This is an intended outcome, not a supervisor crash.

The current user request authorizes unattended chaining through these stages.
It supersedes the historical Plan 16/17 requirement for another launch request
between stages. It does not waive their corrected gates or alter Plan 18 epochs,
objective weights, utility weights, candidate limits or harm budgets.

## Data, source and reproducibility

- Source before optimization and sealed final source are retained separately as
  `source-before/` and `source/`. Final Python source SHA256:
  `1e1e44379439670cbeec9334d051f1b8400ccde7bd93a7964c44ea5803aa8261`.
- Workspace Git HEAD is `ac97de8a60ac4e396ca6ffd8ecfb153bf942994e`; pre-existing
  uncommitted work is preserved. `launch.json` records the dirty workspace,
  actual source/config identities and365 pinned files. A dirty snapshot is not
  represented as a clean Git experiment.
- Initialization is the selection-free final KD checkpoint, SHA256
  `f77547ef2cdc3d5030d537d83a203c0d9f8cf3978020c62a370af78d36cedd62`.
- The entire 13,150-frame training manifest was byte-verified/staged to the
  existing external Linux cache in 107.31 seconds, including all unchanged
  manifest members. The 64 declared development frames were staged separately.
  No RGB/GT was decoded to choose or alter sample IDs, and no samples were skipped.
- `cohort/worklist.json`, `partition.json`, and `data-source-index.json` seal
  240 TRAIN / 64 development frames and 1,539 source files. The keyframe split
  is 7 FIT / 4 VALIDATION / 4 CALIBRATION; partition seed0 is independent of
  fit seeds0/1/2. Partition SHA256:
  `078b2014eab2c800095365529343a078ad14bcc07d567190ac27535c74868b5d`.
- Automatic support-grid candidates are the existing Plan 18 producer: four
  spatially spread 32×32 patches, up to4,096 pixels/1,024 primitives and bounded
  read support. No operator ROI, GT-based region selection or new backbone.
- Physical GPU0 is pinned by UUID. The unrelated GPU1 trainer is preserved.
  WSL monitor polling was paused during GPU work to avoid driver-query stalls;
  matching monitor processes are restored when the coordinator finishes/fails.

## Validation and operational safeguards

**CPU validation passed:** 491 tests, 13 skipped, 0 failed, in121.76 seconds.
Coverage includes routing/gates/unknowns/operating points, CLI runtime, frozen
ownership, checkpoint import, student architecture, native-evaluation wiring,
physical metrics, split roles and data adapters. The 13 skips explicitly require
CUDA input preparation; they are not counted as GPU passes. JUnit evidence is
retained as `validation-cpu.xml` under the execution root after handoff.

The large GroupNorm approximation passed bounded CPU formula/gradient checks but
failed the real frozen-model tolerance and was removed. Those temporary tests
are not represented as production acceptance. The corrected CUDA 512-thread
trial passed its three-shape bitwise check but was not promoted for performance.

Native runtime regression and pilot acceptance must be recorded in
`runtime-validation-safe/result.json` and `pilot-frozen-verification.json`.
Two early regression harness setup errors (archived annotation binding and a
missing required geometry argument) were corrected; failed directories were
retained. The corrected harness calls canonical benchmark metrics directly.
Neither error was reported as a model failure or a passed validation.

The regression and pilot subsequently passed. Native readiness and the full
epoch share one CUDA process lifetime. An earlier harness failure left a CUDA
cleanup thread waiting in WSL's `dxgvmb_send_sync_msg`; a new process started
before that cleanup completed waited at `dxgglobal_acquire_process_adapter_lock`.
The two diagnostic processes were allowed to exit/cancel and reaped before the
detached run. No GPU reset, WSL shutdown or interruption of the unrelated GPU1
training was used. Similar waits are reported in the
[Microsoft WSL issue tracker](https://github.com/microsoft/WSL/issues/41701);
that report is corroborating context, not proof of an identical root cause here.

The detached supervisor checks sealed assets and requires native readiness
reports before the full epoch. It records stage status, command logs, action
checkpoints, immutable gate evidence and terminal summaries. Stage A has bounded
retries for explicit OOM/allocation/I/O errors using strict safe-checkpoint
resume. Numerical/contract errors stop; they are not hidden by restarting or
relabeling data. The last three checkpoints and1,000-sample milestones are kept,
with a4 GiB disk reserve before launching a stage.

Stable handoff requires ≥100 consumed samples, a saved checkpoint, recent finite
losses and exact frozen student/head/model-Adam verification, with both repair
heads updated. Status remains available in `status.json` after Codex exits.
The final coordinator writes `terminal-report.md`, including gate outcomes and
artifact paths, even for a recorded NO-GO. Datasets4/5 and final_test8/9 remain
excluded; non-inferiority margins, multi-seed students and final paper claims
remain untested/unapproved.

## Bàn giao đã nghiệm thu — 19:34 ngày28/10/2026 (UTC+7)

**Stage A đang chạy ổn định trên GPU0**. Lượt audit ghi nhận228/13.150 mẫu đã
tiêu thụ; checkpoint100 và200 đã được lưu. Supervisor và trainer nằm dưới tmux
`frozen-actions`, độc lập với Codex. Việc kết thúc phiên Codex không gửi tín hiệu
dừng cho job này.

Bằng chứng bàn giao:

- `runtime-validation-safe/result.json`:56 frame, disparity/ảnh bitwise bằng
  control, toàn bộ8 metric kiểm tra sai lệch0; native cũng khớp benchmark cũ.
- `pilot-frozen-verification.json`:3 update, cả hai repair cập nhật; student/head/
  model Adam giữ nguyên; model optimizer update0.
- `STABILITY.json`: checkpoint100, student/head/model Adam bitwise giữ nguyên;
  stereo repair100 update, Gaussian repair100 update, model0 update. Selector
  tạm Stage A có252 update; **đây không phải selector cuối Stage B**.
- `HANDOFF_VERIFICATION.json`:365 asset pin được kiểm tra lại;100 log gần nhất
  có loss/reconstruction/repair/selector/KD hữu hạn; tensor checkpoint model,
  actions và selector đều hữu hạn. Scalar gradient-norm không có trong log này,
  vì vậy không ghi là đã đo được norm gradient trong audit.
- `stability/checkpoint-100.pt` giữ riêng để kiểm tra về sau, không bị pruning;
  SHA256 `9c6ed5f2b44d59dc97d4a7344487ca5f24c058b008112d049827d0d0bd16ad51`.
- `validation-cpu.xml`:491 passed,13 skipped CUDA,0 failed; Ruff và
  `git diff --check` đạt. Source/config đang chạy không được sửa sau launch.

Nhịp50 mẫu gần nhất có median1,773 giây/mẫu. Ước tính ban đầu Stage A còn khoảng
6,36 giờ tại mốc audit; con số này không gồm các pha sau, contention hay driver
stall. Không dùng thời gian train để suy FPS inference.

**Chưa hoàn tất:** full1 epoch; freeze/đánh giá repair cuối; Gate A; collect;
Gate B; fit/calibrate/native-evaluate selector. Các pha này đã được nối trong
supervisor theo đúng điều kiện gate. Không báo là selector cuối đã train hoặc
gate nghiên cứu đã pass. Khi chuỗi kết thúc hoặc có NO-GO/failure, xem
`terminal-report.md` và các JSON gate tương ứng trong execution root.

Đọc tiến độ: [status.json](/home/bkai/.local/share/reliableendo-gs/runs/phase2-frozen-actions-20261028T114155Z/status.json).
Đọc bằng chứng: [HANDOFF_VERIFICATION.json](/home/bkai/.local/share/reliableendo-gs/runs/phase2-frozen-actions-20261028T114155Z/HANDOFF_VERIFICATION.json).

Kiểm tra cuối lúc19:38:55 cùng ngày: trainer vẫn chạy, đạt**368/13.150 mẫu**,
checkpoint**300** đã lưu và recent loss hữu hạn. Mốc audit228 và checkpoint100
ở trên được giữ nguyên như bằng chứng lịch sử, không sửa lại theo tiến độ mới.

## Kết quả cuối — kiểm tra ngày29/10/2026

**Stage A đã hoàn tất đủ1 epoch,13.150/13.150 mẫu. Chuỗi dừng đúng protocol tại
Gate A: NO-GO lúc03:28:45 ngày29/10/2026 (UTC+7). Selector Stage B chưa được fit;
Gate B chưa được đánh giá.** Đây là kết quả âm tính của gate, không phải bằng
chứng hoàn thành toàn bộ các pha hoặc lý do để tự chạy lại/đổi threshold.

`stage-a-attempt-00/result.json` ghi `schedule_complete=true`, epoch1,
13.150 optimizer steps. Có `final.pt` và `sample-000013150.pt`.
`stage-a-frozen-verification.json` ghi `status=passed`: student/head/model Adam
giữ nguyên và hai repair head đã thay đổi sau training.
Hash checkpoint cuối đã được đối chiếu lại:
`ef7fa35f49e1ce21bd5994dedcb93b638750f98ca21afa68f37d1feb24fb4ba4`.
Mỗi repair có13.150 update, model có0 update. Selector tạm của Stage A có48.924
update và được loại bỏ; số này không phải kết quả fit selector Stage B.

Kết quả Gate A từ64 frame development đã niêm phong:

| Action cố định | Mean gain PSNR trái, dB | 95% CI theo keyframe | Violation fraction trên outcome đã biết | Can thiệp unknown | Gate |
| --- | ---: | --- | ---: | ---: | --- |
| Stereo repair | -0,000911 | [-0,001602; -0,000215] | 58,33% (21/36) | 27 | Fail |
| Gaussian repair | +0,004223 | [+0,002036; +0,005926] | 19,44% (7/36) | 27 | Fail |

Mỗi fixed policy có63 can thiệp thực thi:36 known và27 unknown; một frame chọn
STOP. Hash checkpoint và64-record native evaluation đã được đối chiếu với
metadata và bằng chứng Gate A; `weights_unchanged=true`.

Gaussian repair có gain dương ở cả4 keyframe và CI dương, violation trên outcome
đã biết nằm dưới ceiling25%. Tuy nhiên vẫn có27 can thiệp chưa xác định được
đầy đủ metric/feasibility, nên không đạt điều kiện không selected-unknown của
gate đã duyệt. Không được coi unknown là safe. Stereo repair còn có mean gain
PSNR trái âm và violation vượt ceiling. Chưa suy nguyên nhân của unknown từ các
con số này; cần audit riêng nếu người dùng yêu cầu bước tiếp theo.

Preflight/cost calibration cho artifact frozen đã được chuẩn bị, nhưng outcome
collection/selector fitting phụ thuộc Gate A không được chạy tiếp. Không đổi
worklist, bỏ frame unknown, chọn lại threshold hay retune trên development để
ép gate pass. Không mở final_test.

Artifact kiểm tra:

- [Gate A](/home/bkai/.local/share/reliableendo-gs/runs/phase2-frozen-actions-20261028T114155Z/gate-a/gate_a.json).
- [Final Stage A checkpoint](/home/bkai/.local/share/reliableendo-gs/runs/phase2-frozen-actions-20261028T114155Z/stage-a-attempt-00/final.pt).
- [Frozen verification](/home/bkai/.local/share/reliableendo-gs/runs/phase2-frozen-actions-20261028T114155Z/stage-a-frozen-verification.json).
- [Terminal report](/home/bkai/.local/share/reliableendo-gs/runs/phase2-frozen-actions-20261028T114155Z/terminal-report.md).

Lệnh đối chiếu PID trên host trong lần kiểm tra này bị bộ duyệt tự động từ chối
vì `workspace routing discovery timed out`; không ghi đó là lỗi trainer và không
khẳng định đã kiểm tra PID trực tiếp thành công. Mốc hoàn tất/NO-GO ở trên được
xác nhận từ các artifact và log đã ghi.
