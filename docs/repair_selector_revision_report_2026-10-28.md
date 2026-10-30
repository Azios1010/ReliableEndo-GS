# Báo cáo sửa frozen repair / selector — 2026-10-28

Người dùng đã cho phép triển khai đề xuất và yêu cầu report kiểm tra được từng
thay đổi. Protocol mới ở [Plan 18](../plans/18_frozen_repair_selector_corrections.md),
cách chạy ở [runbook](frozen_selector_study.md). Đây là đường **phát triển có gate**;
chưa phải bằng chứng Gate I/II hay kết quả tốt hơn cho bài báo.

## Baseline

Đầu lượt không có trainer/supervisor trên host. Log cũ dừng ở stage 1 update
**58.146**, checkpoint cuối **58.000**. Sau khi kiểm tra khóa chống chạy trùng,
dung lượng và hash launch, đã chạy lại đúng supervisor lúc **05:02:26 UTC,
2026-10-28**. Trainer xác nhận resume checkpoint 58.000; 146 update chưa lưu chạy
lại. Đã quan sát log tăng qua **70.107**; mốc cuối lượt nằm trong manifest nghiệm thu.

Baseline dùng source snapshot external `io-continuation-20261028T024455Z/source-buffer`,
tách khỏi workspace đang sửa. Trainer SHA-256 vẫn là
`9e15717b58bfe80049e483c44eb32f5b2a823da99ae5983a507fcaeda1d2569e`.
Giữ 100.100 update stage 1 rồi 10.000 stage 2, cùng model/loss/precision/split/order/
optimizer/scheduler. Không sửa checkpoint hoặc metrics cũ trong lượt này.

## Đã sửa gì

| Vấn đề | Sửa đổi và code chính |
| --- | --- |
| Split ép cả 7 dataset | Validator theo đúng fold có trong manifest; nhận train_1236/dev_7, vẫn chặn overlap, thiếu fold và dataset 8/9. `data/selector_cohort.py`, `config/selector_refit.py` |
| Seed fit làm đổi partition | Tách `partition_seed`/`fit_seed`; giữ hash mặc định lịch sử. `config/selector_refit.py`, `training/selector_fitting.py` |
| Stage A grid không nối được masked selector | Adapter weights-only mới kiểm tra construction/routing/repair forward, không sửa checkpoint và không nới strict resume. `training/student_checkpoint.py` |
| ROI cần operator | `--study-protocol-path` cho Stage A, collection và evaluation; một producer dùng chung cho train/infer/profile. `routing/support_grid.py`, `training/student_runtime.py`, `training/student_learning.py` |
| Token width sai | Hai config frozen chuyển 32→64; fit-many lấy C từ feature bank, kiểm tra dimension. Config frozen token/signed và `training/selector_study.py` |
| Identity bị phụ thuộc selector | Tách actor weights/construction/execution code/renderer khỏi topology/utility/fit seed; cost và selector phải khớp candidate. `routing/artifact_identity.py`, `training/student_learning.py` |
| Không collect-once/fit-many được | Lưu 61 statistics + ba token pool width C trước GT; giữ 12 delta vật lý, timing, unknown/unavailable; xác minh từng shard trước refit. `training/selector_outcomes.py`, `training/selector_shards.py` |
| Utility lệch mục tiêu render | Arm `render_psnr` = mean delta PSNR hai view, đơn vị dB; giữ physical guards và geometry control. Cost chỉ trừ một lần ở decision. `training/metric_quality.py` |
| Chưa orchestration/gate | C0 statistics/union, C1 tokens/local-global, C2 signed/local-global × seed 0/1/2 × hai utility; Gate A/B chặn trước optimizer. `training/selector_study.py`, `evaluation/selector_evidence.py` |
| Cutoff đặt tay | Chọn trên CAL theo bốn harm budget, cap bằng best fixed; thiếu support hoặc selected outcome unknown thì abstain. `training/selector_operating_points.py`, `routing/calibrated_selector.py` |
| Thiếu nghiệm thu | LPIPS hai view với weights pin; CI theo keyframe; promotion cần ≥2/3 seed, đủ cùng cohort và không tăng violation. `evaluation/intervention_selection.py`, `evaluation/selector_promotion.py` |

ROI mới là **candidate sampler**, không phải tissue segmentation: patch 32×32 trải
trên hai trục không gian, tối đa 4 patch/4.096 pixel/1.024 primitive, giao support
deployment/prediction. Không dùng GT/residual để chọn patch. Pooling đếm theo tile
một lần, chỉ materialize patch được chọn. Coverage này có giới hạn và phải được báo cáo.

## Những thay đổi protocol được ghi trước outcome mới

- Stage A có thể train cùng ROI tự động với Stage B. Grid artifact cũ chỉ được nhập
  read-only tường minh; phải đánh giá Gate A lại trên đúng candidate protocol.
- Gate A dùng bootstrap **keyframe**, sửa draft Plan 16 dùng frame: ít nhất một
  repair tăng PSNR trái với lower CI 95% >0, dương ở ≥3/4 keyframe, violation ≤25%,
  không selected outcome unknown. Không coi frame kề nhau là đơn vị độc lập.
- Gate B chỉ dùng FIT: lower CI oracle−best fixed >0; mỗi repair có đóng góp dương
  duy nhất ở ≥2 keyframe. Bỏ quota mỗi action thắng ≥10% của draft Plan 17.
  Best fixed được chọn lại trong từng bootstrap resample. So cùng complete frames;
  báo cáo rõ số frame thiếu outcome bị loại.
- Variant chọn bằng VALIDATION, trung bình đủ ba seed; tie order khai báo C1/C0/C2.
  CAL chọn cutoff cho budget 1/5/10/20%. Không dùng val_7 để chọn variant/cutoff.
  Các điểm development là khám phá, chưa điều chỉnh multiplicity.
- `render_cost_weight_per_ms` mặc định **0**; replay hard budget **1.000 ms cho repair**
  là tham số phát triển, không phải target high FPS. Không tuyên bố đã tìm λ tối ưu.
  Label chỉ chứa delta chất lượng; cost penalty có một owner tại decision.

## Bằng chứng kiểm tra

- Manifest thật, không decode RGB/GT: train **13.150**, dev **4.056**; cohort mặc định
  **240/64 frame**, đúng **7 FIT / 4 VAL / 4 CAL keyframe**. Không đổi split gốc.
- Kiểm thử dùng model/actions/optimizer CPU thật và renderer tổng hợp: frozen-state
  invariance, import, feature bank trước GT, ba arm/ba seed dùng chung store/partition,
  unknown kể cả store hoàn toàn unknown, NO-GO không gọi fit, CAL abstain và một seed
  tốt không đủ promote. Đường CLI Stage A automatic và native evaluator cũng được kiểm tra.
- Wiring LPIPS dùng metric giả trong integration test; không gọi đó là LPIPS evidence.
  Constructor thật vẫn từ chối weights thiếu/ngẫu nhiên/không khớp hash.
- Validation cuối: **486 passed, 13 skipped, 0 failed** (138,36 giây), toàn bộ
  `tests/routing`, các kiểm tra runtime/freeze, metric/native evaluation và A00 adapter.
  Các skip đều là test CUDA/input CUDA khi cố ý tắt GPU cho lượt validation này.
  SHA trước/sau từng file nằm trong
  [manifest nghiệm thu](repair_selector_revision_manifest_2026-10-28.json).
  Patch riêng được lưu vì workspace đã có nhiều thay đổi trước lượt này.
  [Patch riêng](/home/bkai/.local/share/reliableendo-gs/reports/repair-selector-revision-20261028T055433Z/changes.patch),
  JUnit XML và bản file trước/sau nằm trong cùng thư mục artifact bên ngoài repo.
- Lượt rộng đầu: **474 passed, 13 skipped, 10 failed**. Tám failure là fixture cũ
  quên copy `action_update_counts`, đã tái hiện trên source trước sửa; sửa fixture,
  giữ validator nghiêm ngặt. Hai expectation cập nhật cho nullable seed và width 64;
  hash lịch sử được assert bằng giá trị cố định. Sau sửa, bộ focused **74 passed**.

Lệnh validation cuối (CPU, `PYTHONPATH=src`, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`,
`CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`):

```bash
python -m pytest -q -p no:cacheprovider --tb=short \
  tests/routing tests/training/test_student_runtime.py \
  tests/training/test_student_freeze_ablation.py tests/unit/test_intervention_config_identity.py \
  tests/evaluation/test_mask_validation.py tests/evaluation/test_benchmark.py \
  tests/data/test_scared_a00_adapter.py tests/data/test_scared_a00_batch.py \
  tests/data/test_scared_a00_device.py
```

## Chưa chạy / giới hạn / bước tiếp theo

- **Chưa chạy:** full Stage A mới, collect outcome thật mới, fit selector thật trên
  KD artifact. Artifact Stage A có từ trước chỉ là bounded smoke và bị chặn dùng
  như full artifact. Không có gate nghiên cứu nào được ghi là đã pass nhờ test tổng hợp.
- **Chưa kiểm thử:** compiled CUDA renderer end-to-end cho protocol mới; LPIPS native
  với weights hợp lệ. Test CUDA skip được liệt kê và không tính thành pass.
- **Chưa đo FPS:** giữ yêu cầu hoãn trước đó. Replay/action cost không bao gồm đầy đủ
  base, region extraction, selector, render, I/O và contention; không được gọi là FPS.
- **Giới hạn:** student đã thấy FIT/VAL/CAL; chỉ một student seed; ba selector seed
  không loại bias này. CAL ít keyframe và empirical harm budget không là safety guarantee
  dưới distribution shift. ROI nhỏ có thể bỏ sót vùng có lợi.
- **Cần thực nghiệm:** A0/check artifact → train Stage A → Gate A → frozen collection
  → Gate B → fit/calibrate → native development/promotion. Gate fail thì giữ negative
  evidence và pivot; không retune để đạt gate. Latency chỉ đo khi được yêu cầu tiếp.
- **Chưa mở:** final_test 8/9; chưa chọn non-inferiority margins; chưa có claim paper
  thắng baseline. Không bảo đảm kết quả tốt chỉ vì code đã qua kiểm thử.

## Nguồn workflow

Appraisal dùng skill `scientific-critical-thinking`: Kassis, T., Agarwal, V., He, Y.,
Patel, D., & Brueckner, A. M. (2026),
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065).
Record hiện hành đã kiểm tra ngày 2026-10-28 (v2); đây là nguồn workflow, không là
bằng chứng hiệu quả cho ReliableEndo-GS.
