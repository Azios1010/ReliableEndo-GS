# Audit selector, mask, cost và utility — 2026-10-07

## 1. Phạm vi và trạng thái

Báo cáo này ghi lại toàn bộ vấn đề tìm thấy trong audit nhánh selector dùng mask
và pilot joint đã hoàn tất. Đây là bằng chứng chẩn đoán của một thử nghiệm nhỏ;
không phải kết quả final test hoặc chứng nhận contribution.

- Dữ liệu active: original SCARED, dataset_1–dataset_7.
- Pilot: 56 frame TRAIN, 28 frame development, hai keyframe mỗi dataset.
- Development khác frame nhưng cùng keyframe với TRAIN.
- Khởi tạo student/head từ checkpoint quarter-run đã có; 56 frame là worklist
  cập nhật thêm, không phải toàn bộ dữ liệu mô hình từng thấy.
- Lịch chạy: 3 joint epoch, 168/168 optimizer updates; exit code 0.
- Student, Gaussian head, stereo repair, Gaussian repair và selector đều có
  nonzero parameter update ở mọi bước của pilot.
- KD lấy từ cache read-only; không chạy teacher inference trong pilot.
- Evaluation giữ nguyên weights; không cập nhật normalization và bảo toàn RNG.
- GT chỉ được đọc sau proposal, quyết định và thực thi counterfactual action;
  không đưa GT vào deployment input hoặc feature của selector.
- Gate II chưa đạt. Chưa chứng minh lợi ích dual-source selector.

Lượt audit sau pilot chỉ đọc code/artifact, tính thống kê CPU, kiểm tra mask và
replay allocator trên predictions/labels đã lưu. Không sửa thuật toán, config,
checkpoint hoặc raw metrics; không train thêm.

## 2. Kết quả pilot cần phân biệt

| Trạng thái | Stereo depth MAE, mm ↓ | Stereo depth RMSE, mm ↓ | PSNR trái, dB ↑ | PSNR phải, dB ↑ |
| --- | ---: | ---: | ---: | ---: |
| Trước joint, STOP | 10.622639 | 16.244039 | 19.412466 | 19.101204 |
| Sau joint, STOP | 9.159580 | 14.903522 | 19.622751 | 19.313140 |
| Sau joint, selector | 9.159580 | 14.903522 | 19.655141 | 19.325695 |
| Sau joint, Gaussian cố định | 9.159580 | 14.903522 | 19.655726 | 19.329798 |

Depth macro means có 27 mẫu hợp lệ; render macro means có 28 mẫu.

Joint làm giảm MAE trung bình 1.463059 mm, khoảng 13.77%. MAE giảm ở 14/27 mẫu
và tăng ở 13/27 mẫu. PSNR tăng trung bình khoảng 0.21 dB mỗi view; normal
consistency giảm nhẹ từ 0.555865 xuống 0.548618. Chất lượng chưa cải thiện đồng
đều trên mọi mẫu hoặc mọi geometry metric.

Selector chọn Gaussian ở 28/28 mẫu. Phần tăng thêm do intervention chỉ khoảng
0.032390 dB trái và 0.012555 dB phải so với STOP sau joint. Gaussian không đổi
stereo disparity, nên mức giảm stereo-depth MAE trong bảng đến từ joint base
learning; không phải bằng chứng selector giảm depth MAE. Stereo depth và hình
học Gaussian/rendered depth là những phép đo khác nhau.

## 3. Vấn đề đã xác nhận

### F01 — HIGH: Cost estimate giữ outlier vĩnh viễn

**Cơ chế:** `balanced_learning.py` cập nhật `cost_max_ms` bằng maximum giữa
estimate cũ và mọi timing mới. Một quan sát chậm ảnh hưởng mọi quyết định sau đó.

**Bằng chứng:**

- Stereo peak 2635.552710 ms ở update 26.
- Checkpoint lưu estimate 2635.552734 ms.
- Total budget 1000 ms; remaining development budget khoảng 660–863 ms.
- Stereo bị loại trên toàn bộ candidate trước khi action selection.
- Development counterfactual stereo thực đo 102.328–113.263 ms, mean 108.912 ms.

`BalancedHarness.collect()` đo synchronized action execution dưới `no_grad`.
Backward không nằm trong cửa sổ timing này. Nguyên nhân outlier chưa được truy
bằng trace; không được kết luận do backward, OS kill hoặc CUDA crash.

**Thiếu:** calibration inference riêng, warmup theo shape, đo lặp, nhận diện
outlier/tail và provenance của mask/halo/hardware/debug mode.

**Hướng xử lý:** đo trên TRAIN, lưu phân phối và overrun evidence; tách estimate
phục vụ deployment khỏi maximum lịch sử. Không thay estimate bằng timing từ
held-out development rồi coi đó là calibration đã kiểm chứng.

### F02 — HIGH: Ranking gần như không nhận supervision

`BalancedSelector.objective()` áp dụng `ranking_margin = 0.02` lên chênh lệch
**raw global utility**, trước bước normalized weighting.

- Utility thường ở mức 0.0001–0.003 trong chế độ sửa bằng mask.
- Chỉ 2/570 cặp TRAIN vượt margin.
- Ranking loss bằng 0 ở 167/168 bước.

Tăng `ranking_weight` không tạo gradient cho các cặp đã bị loại bởi margin.

**Hướng xử lý:** calibrate margin theo scale/noise của TRAIN, ghi tỷ lệ informative
pairs, giữ semantics raw physical guards riêng. Không chọn margin trên development
hoặc nhân reward nhằm ép action diversity.

### F03 — HIGH: Utility supervision chưa khớp mục tiêu geometry/render

Nhánh đang chạy dựng `BalancedSelector` và `BalancedHarness`.
`component_targets()` tính utility từ:

- global RGB improvement;
- global warp photometric improvement;
- outside collateral penalty.

Local RGB/warp tạo local auxiliary target. MAE/RMSE/p95 và PSNR/SSIM trong
`metric_diagnostics` chỉ dùng để chấm kết quả; chưa tạo supervision cho selector
trong masked joint path.

`OutcomePredictor` và canonical metric tradeoff đã tồn tại trong repository,
nhưng chưa nối vào path này. Có metric harness không đồng nghĩa MLP đã học
metric utility.

**Hướng xử lý:** dự đoán các metric changes riêng, common TRAIN-only scales và
utility/guards từ một canonical owner. GT chỉ có thể là detached offline TRAIN
supervision; không thành deployment input, proposal hoặc feature. Joint actions
thay đổi cần target mới có identity; không dùng stale frozen-action labels.

### F04 — HIGH: Proxy harm và metric violation là hai sự kiện khác nhau

Proxy harm là bất kỳ normalized local/global/outside RGB/warp regression nào
vượt tolerance 0.001. Metric guard kiểm tra depth, render, coverage và collateral
trong raw units tương ứng.

Trong 29 vùng stereo đủ metric:

| Nhãn proxy | Metric safe | Metric violation |
| --- | ---: | ---: |
| Proxy harmful | 1 | 12 |
| Proxy safe | 10 | 6 |

Có 7/29 vùng bất đồng. Risk predictor hoàn hảo theo proxy vẫn có thể bỏ lọt
metric violation hoặc từ chối metric-safe action.

**Hướng xử lý:** xác định đúng event cần bảo vệ, học joint/component risk cho
canonical metric guards và xử lý missing observations rõ ràng. Calibration
không sửa được việc chọn sai event.

### F05 — HIGH: Risk prediction chưa đáng tin

| Chỉ số trên 30 candidate regions | Stereo | Gaussian |
| --- | ---: | ---: |
| Observed proxy-harm count | 14 | 3 |
| Mean predicted proxy risk | 0.826044 | 0.088571 |
| Brier score | 0.360361 | 0.108373 |
| Proxy-risk AUC | 0.638393 | 0.333333 |
| False positives tại cutoff 0.5 | 16 | 0 |
| False negatives tại cutoff 0.5 | 0 | 3 |

Stereo prediction 0.579768–0.969074 khiến mọi candidate bị từ chối. Gaussian
predictions đều dưới cutoff 0.5, bao gồm ba candidate harmful. Đây là thống kê
nhỏ trên development cùng keyframe, chưa phải reliability calibration độc lập.

TRAIN stereo proxy-harm rates cũng khác development: 43/64, 46/63, 43/63 qua
ba epoch, so với 14/30 trên development. Data/target shift và khả năng tổng quát
cần được kiểm tra; không mặc định lỗi chỉ nằm ở threshold.

**Hướng xử lý:** calibration trên nhóm TRAIN fit/calibration tách biệt, đúng risk
event, giữ unknown/abstention; không hạ risk cutoff để buộc stereo được chọn.

### F06 — HIGH: Sửa cost/risk gate chưa làm utility head chọn stereo

CPU replay dùng canonical `select_repairability()` trên predictions đã lưu:

| Scenario chẩn đoán | Action được chọn |
| --- | --- |
| Cost đã ghi trong checkpoint | Gaussian 28/28 |
| Cost giả định stereo 110 ms, Gaussian 50 ms | Gaussian 28/28 |
| Cost giả định trên và bỏ risk gate | Gaussian 28/28 |

Hai scenario sau là intervention giả định để cô lập decision logic; không phải
calibration đã triển khai hoặc performance experiment.

Predicted global gains trên toàn bộ candidate:

- Stereo: -0.001021 đến +0.000212.
- Gaussian: +0.000602 đến +0.001955.

Vì Gaussian đứng trên stereo trong predicted gain ordering, sửa cost alone chưa
đủ. Replay này không chứng minh kiến trúc không thể học; nó mô tả decision
surface của checkpoint hiện tại.

### F07 — MEDIUM: Local/global gradients vẫn xung đột

Hai gain outputs dùng chung encoder. Harm có encoder riêng, nên harm BCE không
trực tiếp lấn át gain-network gradients qua shared parameters.

- Raw local/global gradient-norm ratio median khoảng 22.79.
- Sau actual recorded weighting: median khoảng 1.60, maximum khoảng 83.99.
- Negative local/global gradient cosine ở 50/168 bước.
- Local balancing coefficient chạm floor 0.05 ở 55 bước.

Normalization/balancing đã có nhưng chưa bảo đảm directional agreement hoặc
stable global ranking. Không có cơ sở cho một hệ số cố định 1000x.

**Hướng xử lý:** kiểm tra contribution sau weighting và clipping, gradient
conflict, scales/common units và ablation tách head/auxiliary influence.

### F08 — MEDIUM: Local utility chưa trực tiếp tham gia decision

`BalancedSelector.predict()` chỉ trả global gain và harm. Local gain là
auxiliary output, không được dùng trực tiếp để chọn action. Đây là formulation
hiện tại, không phải lỗi vô tình bỏ tensor.

Budget tối đa 4096 pixel trên grid 512×640 chỉ khoảng 1.25% ảnh. Whole-frame
reductions có thể làm nhỏ tín hiệu sửa cục bộ. Trong counterfactual stereo,
mean fixed-support depth gain global khoảng 0.002397 mm, local khoảng 0.168800 mm.

**Hướng xử lý:** kiểm tra local/global physical utility và guards nhất quán;
không cộng cùng render reward hai lần hoặc ưu tiên local gain mà bỏ global harm.

### F09 — HIGH về evidence: Chưa chứng minh selector tốt hơn Gaussian-only

- Selector Gaussian 28/28; không có learned action switching trong pilot.
- Selector có 3 frame metric violations và 1 unknown metric intervention.
- Fixed Gaussian có 2 violations và 1 unknown; PSNR trung bình nhỉnh hơn selector.
- Stereo/Gaussian predicted-region baselines giúp tách region choice và action
  choice, nhưng stereo baselines trong budget hiện tại đều rơi về STOP.

Không dùng kết quả này để tuyên bố superiority, dual-source contribution hoặc
Gate II pass.

## 4. Vấn đề cần kiểm chứng thêm

### F10 — MEDIUM, suspected: Proposal saturation có thể gây spatial bias

`candidate_scores()` clamp warp/render proxies về [0,1]. Khi vượt scale, severity
ordering bị mất. `argsort(..., stable=True)` giữ flattened order của các pixel
đồng hạng; budget truncation có thể ưu tiên upper/left scanline.

Trong pilot:

- 27/30 region masks chạm mép trên input ROI.
- Median selected vertical centroid ở khoảng 11.06% chiều cao ROI.
- Bbox fill fraction thay đổi khoảng 0.159–0.838.

Đây là evidence về spatial concentration, chưa chứng minh saturation là nguyên
nhân. Raw per-pixel score maps/tie counts chưa được ghi; image errors thực cũng
có thể tập trung ở đầu ROI.

**Hướng kiểm tra:** score trước clamp, tie fraction, spatial coverage và bounded
read-box cost; thay đổi ranking chỉ khi có targeted evidence.

**Đã xác nhận tốt:** cả 30 stored masks có unique indices và không ghi pixel ra
ngoài input ROI đã khai báo. ROI inclusion không chứng minh exact foreground.

### F11 — MEDIUM, dynamics: Normalization chuyển trạng thái giữa lúc học

32 snapshot đầu dùng raw features trước khi statistics fit. Sau đó chuyển sang
normalization đã fit, trong khi student/head/actions tiếp tục cập nhật và
statistics giữ nguyên.

Điều này có thể gây input/target distribution shift. Feature clipping trên các
TRAIN epoch sau chỉ khoảng 1%, nên chưa có evidence rằng toàn bộ feature vectors
bị clipping. Chưa xác nhận normalization dynamics là nguyên nhân chính.

**Hướng kiểm tra:** controlled normalization/burn-in ablation và drift logging
trên TRAIN, checkpoint đầy đủ statistics. Không dùng development để refit.

### F12 — Chưa chứng minh: Student/feature thiếu thông tin

Feature hiện có RGB/warp, matching entropy/margin, geometry, Gaussian/render và
local/global summaries. Có thể cần mask shape, read-area/cost context hoặc tail
residual statistics, nhưng dữ liệu hiện tại chưa chứng minh thông tin thiếu là
nguyên nhân chính.

Ưu tiên sửa supervision/calibration đã xác nhận trước; thêm feature hoặc tăng
MLP size cần controlled ablation. Mean/std summaries không tự chứng minh đủ
thông tin về repairability hoặc calibrated uncertainty.

## 5. Giới hạn dữ liệu, metric và claim

### F13 — Một development frame thiếu GT

Một frame có zero GT-valid pixels. Depth metrics chỉ có 27/28 frames. Unknown
không được đổi thành safe, zero error hoặc silently drop khỏi violation accounting.

### F14 — Chưa có rendered-depth metrics

Renderer trong path này chưa cung cấp rendered depth. Rendered-depth MAE/RMSE,
AbsRel và coverage unavailable. Stereo-depth metric không thay thế chúng.

### F15 — Joint improvement chưa đồng đều

MAE giảm trung bình nhưng tăng ở 13/27 frames; normal consistency giảm nhẹ.
Loss tổng giảm không chứng minh mọi geometry/render metric hoặc selector đều tốt.
Cần paired per-frame, per-keyframe results và uncertainty của comparison.

### F16 — Development cùng keyframe, pilot nhỏ

56 TRAIN, 28 development và 3 epoch là debugging evidence. Không có
sequence-disjoint generalization, final-test evidence hoặc confidence intervals
trong báo cáo này. Initial artifact đã có prior TRAIN exposure; không mô tả
pilot như training from scratch trên 56 frames.

### F17 — ROI thủ công, chưa có foreground model

ROI là approximate operator tissue regions chọn từ RGB, có provenance/hashes.
Chúng chưa phải pixelwise tissue segmentation; vùng tối/specular tissue có thể
còn trong ROI. GT/outcomes không được dùng để chọn ROI. Chưa chứng minh một
student hoặc frontend tự tạo đủ foreground/action-mask evidence khi deployment.

### F18 — Sparse writes chưa chứng minh sparse execution hoặc FPS

A03 geometry, field composition, một phần feature extraction và cả hai renders
vẫn chạy full grid. Matching dùng bounded left ROI nhưng right strip giữ full
width để bảo toàn correspondence. Ít selected pixels/primitives chưa đủ chứng
minh latency saving.

Timing pilot dùng `CUDA_LAUNCH_BLOCKING=1`, synchronized single-run observations.
Policy timing không bao gồm đầy đủ I/O hoặc oracle overhead; chưa có
end-to-end latency/FPS benchmark hoặc matched-cost efficiency claim.

### F19 — Metric feasibility chưa phải balanced utility winner

Stereo có 11/29 complete regions vượt guards và có một rewarded component
improvement; Gaussian có 26/29. Đây là counts của candidates, không phải số
frame mà stereo là action tốt nhất theo metric utility.

Khi replay measured PROXY oracle với hypothetical feasible costs, oracle chọn
Gaussian 24 frame, stereo 3 frame, STOP 1 frame. Trong ba stereo choices, hai
vi phạm metric guards. Proxy-oracle diversity không chứng minh physical utility
diversity hoặc calibrated safety.

### F20 — Gate II và threshold evidence còn thiếu

Không có Gate II pass. Guard config là pilot contract; số violation được báo
phải gắn với đúng config identity. Không nới guard/tolerance/risk cutoff để loại
negative evidence. Nếu cần thay physical tolerances, phải có TRAIN provenance,
noise/effect-scale study và comparison mới; development không dùng để chọn
policy thắng, dataset_8/9 không dùng cho tuning.

## 6. Vấn đề cũ và mới

| Nhóm | Trạng thái qua các lượt |
| --- | --- |
| Cost/risk chặn stereo | Đã thấy trước; audit hiện tại truy được mechanism và extent. |
| Proxy utility/risk chưa khớp metric | Vấn đề cũ; metric harness chưa nối vào masked joint supervision. |
| Ranking gần như bằng zero | Mới phát hiện/định lượng trong chế độ mask với margin cũ. |
| Local/global conflict | Cơ chế đã có; pilot mới cho measured norm/cosine evidence. |
| Score saturation/spatial bias | Nghi vấn mới của mask proposal, chưa xác nhận cause. |
| Thiếu student information | Chưa được chứng minh; không coi là confirmed finding. |

Các sửa trước đã làm wiring/smoke và bounded joint run hoạt động, mask writes
đúng ROI; chúng chưa giải quyết hết supervision, calibrated risk hoặc action
selection quality.

## 7. Thứ tự remediation đề xuất — chưa triển khai trong audit

1. Dedicated TRAIN inference-cost calibration: warmed no-grad actions, repetitions,
   mask/halo shapes, hardware/debug provenance, explicit tail/overrun policy.
   Giữ lại maximum/outlier evidence lịch sử.
2. Restore informative ranking trên TRAIN-calibrated utility scales; đo coverage,
   weighted norms/cosines và local/global influence.
3. Connect separate physical metric outcomes và canonical balanced utility/guards
   vào masked joint selector supervision. Reuse metric owners đã có; version
   moving-action targets/checkpoints; giữ GT ngoài deployment inputs/features.
4. Risk event đúng physical guards, joint/component labels, unknown masks và
   TRAIN fit/calibration groups riêng. Không force action diversity.
5. Verify proposal score ties, ROI topology/read cost; chỉ bổ sung feature hoặc
   đổi architecture khi có targeted ablation evidence.
6. Rerun bounded immutable pilot và so STOP, fixed stereo/Gaussian, single-source
   policies, selector và appropriately defined oracle trên cùng development.
   Chỉ tăng quy mô khi các tín hiệu học/quality và stage gates có evidence.

Không cần blind full-dataset retraining để bắt đầu sửa cost hoặc audit saved
predictions. Remediation supervision sẽ cần thử nghiệm mới được định danh.

## 8. Validation và provenance

### Đã chạy trước lượt tổng hợp này

- Native pilot completed: 168 updates, 56 TRAIN/28 development, GPU RTX 5070 Ti,
  PyTorch 2.7.1+cu128, CUDA 12.8.
- CPU asset preflight: 56/56 exact cache hits và 84 ROI hashes/grids.
- Focused regression suite: 198 passed, gồm CPU renderer substitution.
- `mypy src`: success, 206 source files.
- Ruff check/format, CLI help, shell syntax và `git diff --check` passed trong
  lượt chuẩn bị pilot. Đây không phải claim full CI trên publication checkout.
- Read-only CPU statistics, canonical allocator replays và ROI inclusion audit.

### Không chạy trong lượt audit/tổng hợp này

Không train thêm, không teacher inference/cache generation, không native
performance experiment, không sửa thresholds/splits, không final-test evaluation.
CPU substitution không xác nhận CUDA kernel correctness. Native run thành công
là evidence riêng từ host run, không được suy ra từ CPU tests.

### Artifact identity

- Run identity: `mask-validation-bnx61MIx` (external artifacts; machine paths
  được cấu hình khi vận hành, không đưa dataset/checkpoint vào repository).
- Git HEAD của run: `e49efbd293ed9c0bbb569182e2419fcc77b12f0b`, dirty worktree.
- First-party source SHA-256:
  `63b654fdc3b073712af6bce68abea12c6453a4c6383076080d67d997457a76db`.
- TRAIN parent manifest hash:
  `7c46070c9b393dde5afb0af4820715413dd079956f96205affea33261d1827ba`.
- Development parent manifest hash:
  `6a328332026a5b52a936df14e849952cdb25efa5f4e651847b97590b71d1695d`.
- Worklist SHA-256:
  `91e0553056ccccd444b19588f66a78493240f58a8ac547e64a22a17bf5a0a318`.
- ROI index SHA-256:
  `b2a67b426a9d85a37bcfac85fd43ab0f4a42a8c4ad0fcb87fde254a11036772c`.
- Final checkpoint SHA-256:
  `b03e0746482f91be1c3764569202d64388776a8c2f6859aa5efbe14227ec98e7`.
- Intervention config hash:
  `9abdf5e35f91b62a1bb49efa61c0209b0565476d69ed61542f5c4894b1aa5266`.
- Metric guard config hash:
  `fd9ff308bd6b7cd6e2c17b6a5fd7ca161d32f936c6752f83419b6698a08f69fb`.
- Mask config hash:
  `2961ca9e82abd451ea98f72c3fb1095fc3f311314a532bb1e02e1e4074b7118f`.

Raw metric JSONL, checkpoint, cache, ROI arrays và contact sheets ở external
storage. File Markdown chỉ lưu derived findings/provenance; không chứa medical
images hoặc absolute dataset paths. Artifact hashes không chứng minh một commit
chưa được tạo là đã chạy; giữ source hash và dirty-state evidence của run.

### Source owners

- [Cost update và joint loss](../src/reliable_endo_gs/training/balanced_learning.py)
- [Actual shared counterfactual action harness](../src/reliable_endo_gs/training/balanced_interventions.py)
- [Proxy component targets](../src/reliable_endo_gs/training/quality_components.py)
- [Active selector và ranking](../src/reliable_endo_gs/routing/balanced.py)
- [Normalization](../src/reliable_endo_gs/routing/normalization.py)
- [Deployment allocation](../src/reliable_endo_gs/routing/repairability.py)
- [Mask proposals](../src/reliable_endo_gs/routing/masks.py)
- [Metric outcome predictor hiện có](../src/reliable_endo_gs/routing/outcome_predictor.py)
- [Canonical metric utility/guards](../src/reliable_endo_gs/routing/tradeoff.py)
- [Independent mask metric diagnostics](../src/reliable_endo_gs/evaluation/mask_metrics.py)

## 9. Kết luận

Pipeline đã hoàn tất bounded joint run. Contribution của selector chưa được
chứng minh. Cost exclusion, proxy-risk rejection và predicted global-proxy gain
ordering cùng làm Gaussian chiếm toàn bộ lựa chọn. Dead ranking và khác biệt
giữa supervision với intended metric event là các vấn đề actionable.

Chưa có evidence cho kết luận mạng không thể học action hoặc student thiếu
thông tin là nguyên nhân chính. Negative findings, missing metrics và Gate II
failure cần được giữ nguyên khi triển khai remediation tiếp theo.
