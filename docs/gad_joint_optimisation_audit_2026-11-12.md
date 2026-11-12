# GaD: kiểm toán các thí nghiệm joint stereo–Gaussian optimisation

Ngày kiểm toán: 2026-11-12. Repository HEAD: `9da799be696fcf18a2545906badcdf94559d2d31`.

Báo cáo được viết theo yêu cầu bổ sung của người dùng. Các thao tác điều tra chỉ đọc source, cấu hình, Git history, CSV/JSON/log và tài liệu; không chạy training, inference, evaluation hoặc benchmark. Chỉ file báo cáo này được tạo. Working tree đã có thay đổi trước khi kiểm toán; chúng được giữ nguyên.

## A. Executive finding

**Kết luận tổng thể: PARTIALLY CONFIRMED.**

Repository và các artifact bên ngoài được repository tham chiếu **có nhiều thí nghiệm joint stereo–Gaussian training thật**, bao gồm một continuation gây regression geometry và một so sánh joint-versus-frozen student có kết quả bất lợi cho joint ở một số chỉ số. Tuy nhiên, **không có bằng chứng rằng mọi joint configuration đều kém hơn**, và **chưa tìm thấy ablation đối sánh trực tiếp fixed-centre versus trainable-centre attribute refinement**.

Các kết luận cần tách riêng:

| Câu hỏi | Kết luận |
| --- | --- |
| Đã từng joint-train stereo và Gaussian reconstruction chưa? | **CONFIRMED**: source, optimizer, checkpoint metadata và native evaluation còn tồn tại. |
| Có run joint bị kém đi không? | **CONFIRMED trong phạm vi run cụ thể**: continuation 128→1152 tăng depth MAE/RMSE, nhưng right PSNR tăng. |
| Frozen student luôn tốt hơn joint? | **Không được xác nhận**: frozen tốt hơn MAE/EPE/PSNR trong paired pilot, nhưng joint có RMSE thấp hơn nhẹ. |
| Photometric gradient luôn làm hỏng stereo? | **Bị phản chứng bởi các controls hiện có**: KD + photometric có EPE thấp hơn KD-only ở cả ba seed; photometric-only lại collapse. |
| Đã chứng minh trainable Gaussian centres kém hơn fixed centres? | **NOT FOUND**: các run joint tìm được chủ yếu cập nhật centres gián tiếp qua disparity; không phải học centre-offset độc lập. |
| Attribute refinement có thể cải thiện RGB và giữ centres stereo? | **CONFIRMED cho pipeline đã lưu**: code giữ centres/disparity; fixed-centre refinement cải thiện PSNR/SSIM/LPIPS. |
| Đã chứng minh rendered surface geometry không đổi? | **Chưa**: các rendered-depth metrics kiểm tra được đều null; stereo geometry và rendered surface là hai đối tượng khác nhau. |

Kết luận có thể dùng cho GaD: **attribute refinement cải thiện rendering trong khi bảo toàn stereo-derived centres theo cấu trúc mô hình**. Kết luận mạnh hơn rằng **cho centres di chuyển sẽ kém hơn** vẫn cần ablation riêng. [E1–E8]

## B. Experiment inventory

Quy ước: A = joint stereo–Gaussian training; B = refinement cho phép cập nhật centres độc lập cùng attributes; C = fixed-centre attribute refinement; D = controls khác. “Confidence cao” xác nhận artifact và cách diễn giải trong phạm vi ghi rõ, không đồng nghĩa bằng chứng nhân quả hoặc generalization mạnh.

| Experiment | Training strategy | Trainable components | Geometry results | Rendering results | Evidence path | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| SCARED-C Stage1 60k → Stage2 cũ 10k, 2026-09 | A; toàn model, GT disparity + same-view rendering | Stereo backbone và GS regressor; centres theo disparity, không có centre-offset riêng | Validation EPE 1.356977 → 1.326974 px; khác branch đánh giá nên chưa coi là matched improvement | Render L1 gần plateau: 0.290927 tại 1k → 0.290181 tại 10k | E1: CSV + provenance + audit | Cao về số lưu; thấp cho claim metric geometry |
| SCARED-C Stage2 sigmoid 3k | A/D; sửa scale parameterization | Toàn model như trên | EPE 1.344180 px | L1 0.275989; 1−SSIM 0.631777 | E1 | Cao; scheduler khác control |
| `joint-larger-gpu1-4vgk9wi_`, 128→1152 | A; joint student/head/repairs/selector | Student + attribute head; stereo/Gaussian repair; selector có optimizer riêng | MAE 13.238349 → 16.095515 mm; RMSE 39.843979 → 54.940627 mm | PSNR L 19.736927 → 19.659149; R 19.385441 → 19.513953 dB | E2 | Cao; development diagnostic |
| `joint-geometry-remediation-qayuedy8`, control256 vs geometry256 | A/D; thêm direct GT metric-depth objective | Như trên; thêm gradient geometry cho student/stereo repair | MAE 14.349148 → 9.616876 mm; RMSE 44.442064 → 15.029152 mm | R PSNR 19.532353 → 19.434172 dB | E3 | Cao; không phải final test |
| `joint-versus-frozen-gpu0-11fuurbl`, cursor459 | A versus frozen student/head + trainable repairs | Joint: tất cả owners; frozen: student/head và model Adam giữ nguyên, repairs/selector vẫn học | Selected EPE 8.088562 vs 7.495156; MAE 10.436251 vs 9.272147; RMSE 15.645196 vs 15.696837 | R PSNR 19.243779 vs 19.411244 | E4 | Cao; một bounded paired pilot |
| Fresh-context, signed-gain, mask-token pilots | A/D; biến thể selector/loss/feature context | Student/head và repairs học; selector supervision detached | Các số riêng ở C.4; không có direct centre refinement | Có native RGB metrics; không đủ để quy lỗi cho centre movement | E5 | Cao về số; thấp cho decoupling causal claim |
| Three-epoch KD+photometric / KD-only / photometric-only / GT controls | A/D; thay nguồn supervision | KD+RGB, RGB-only, GT+RGB: student và head; KD-only: head không nhận gradient | KD+RGB tốt hơn KD-only nhẹ ở 3 seed; RGB-only EPE 51.653 px | Evaluator của các geometry controls không render; không có RGB score tương ứng | E6 | Cao; 55 GT-valid validation frames |
| Clean 10-epoch student + 1-epoch full refiner | A trước, C sau | Stage đầu: student+head; stage sau: chỉ refiner 1,671 parameters | Pooled EPE 2.102800869 px; mean frame RMSE 5.596403262 mm, base/refined giống nhau | R PSNR pooled 24.103326 → 25.144021 dB | E7 | Cao; main held-out artifacts |
| Full/scale-only/affine/global-scale controls | C/D | Stereo frozen; correction module hoặc global-scale heuristic; không có centre offsets | Stereo metrics bằng nhau trong matched cohort | Full correction tốt hơn identity; global-scale heuristic có right PSNR thấp hơn identity | E8 | Cao; post hoc 296-frame cohort |
| Direct trainable-centre + attribute refinement | B | Chưa tìm thấy implementation/run đối sánh tương ứng | Không có kết quả đáng tin để báo cáo | Không có | Search source/config/history + các roots ở G | Không kết luận vượt quá “not found” |

### B.1. Gradient flow và parameter ownership thực tế

**First-party student training:** `StudentReconstructionModel` đưa disparity dự đoán vào reconstruction mà không detach. Chuỗi disparity → calibrated geometry → prior centres → Gaussian composition → render còn khả vi. `compose_gaussian_field` gán `means3d` từ `prior.mu`; residual head chỉ dự đoán rotation/log-scale/opacity. Vì vậy photometric loss có thể cập nhật stereo và làm centres ở lần forward tiếp theo thay đổi, nhưng không tồn tại một tập free XYZ parameters tương đương case B. [E9]

`StudentLearner` có model Adam chứa student và attribute head; action Adam chứa repair heads; selector Adam riêng. Selector features/labels được detach, nên selector MSE không tự truyền gradient về student. Trong compact mặc định, tài liệu ghi student 12,481 parameters, attribute head 19,207, stereo repair 1,441, Gaussian repair 1,735, selector 1,730. Không áp các counts này cho multiscale main model khoảng 1.45 M parameters. [E9]

**Loss:** simple student stage dùng `reconstruction_weight × Lrgb + kd_weight × Lkd`, joint stage thêm `repair_weight × Lrepair`. `Lrgb` là mean Charbonnier của hai views; KD là masked SmoothL1 giữa disparity student và detached teacher. Các scalar weights mặc định là 1, Adam LR 0.001. Các balanced intervention pilots thực tế còn có warp, repair/local/global terms và adaptive gradient-balance weights; không nên mô tả chúng là đơn giản `RGB + KD` với effective weights cố định. Geometry remediation thêm SmoothL1 của depth error chuẩn hóa 1 mm, configured weight 1. [E3, E9]

**Upstream Stage2:** AdamW dùng `model.parameters()`, loss `Ldisp + 0.8 L1 + 0.2(1−SSIM)`. Training branch giữ flow prediction khả vi và dùng nó để dựng depth/XYZ/GS attributes. Các `.detach()` trong inference branch hoặc `PseudoColor` không phải gradient boundary của Stage2 training. Không được phân loại Stage2 là frozen stereo chỉ vì tên stage. [E1, E9]

**Fixed-centre refinement:** trainer freeze toàn bộ base model và gọi nó dưới `torch.no_grad()`; Adam chỉ sở hữu refiner. Refiner lấy `mu=field.means3d`, thay rotation/scale/opacity, giữ colours trong main configuration và truyền lại base disparity khi render. Head-only control chỉ unfreeze `model.head`, stereo vẫn frozen. Đây là case C. Main refiner dùng cả hai view Charbonnier, LR 0.001, không cần teacher disparity hoặc GT trong refinement objective. [E7–E9]

## C. Recovered numerical comparisons

Các bảng dưới chép scalar đã lưu, làm tròn để đọc; không chạy lại model hoặc tạo metric từ predicted pixels. Artifact JSON/CSV giữ precision gốc. MAE/RMSE đơn vị mm, EPE px, PSNR dB.

### C.1. Joint continuation thật sự có regression, nhưng không ở mọi metric

`joint-larger-gpu1-4vgk9wi_`: cùng 140 diagnostic images, 20/fold từ datasets 1–7, 512×640. Thêm 1,024 FIT samples sau parent128, tổng 1,152 updates; chưa đủ một full epoch của cohort 2,788 FIT entries. Các số sau thuộc policy sau selected action. [E2]

| Metric | Parent128 | Joint1152 |
| --- | ---: | ---: |
| EPE | 7.870063 | 8.107486 |
| Stereo depth MAE | 13.238349 | 16.095515 |
| Stereo depth RMSE | 39.843979 | 54.940627 |
| Gaussian-centres Chamfer | 7.705606 | 9.198204 |
| Left PSNR | 19.736927 | 19.659149 |
| Right PSNR | 19.385441 | 19.513953 |
| Left SSIM | 0.638787 | 0.636528 |
| Right SSIM | 0.628045 | 0.629489 |
| Physical guard violations | 25 | 27 |

STOP-only MAE cũng đã tăng 13.238349 → 16.095515 mm: regression tồn tại ở base, không chỉ ở selector. Selector không chọn stereo trong hai assessments, Gaussian repair không thay stereo disparity. Checkpoint cuối `train/stopped.pt`, SHA-256 `b30bee03c3abf186e32c92c9f63dbe92797634c44d1b82e4e03de0d7540d8e0f`. Training 33.48 phút, peak allocated 5.680 GiB; đây không phải inference latency. Không có LPIPS hoặc rendered-depth error đáng tin trong assessment này. [E2]

### C.2. Paired joint-versus-frozen student

Hai nhánh cùng parent geometry203, cùng 256 FIT entries tiếp theo, cursor203→459; cùng teacher cache, loss, guard policy và cùng 140 diagnostic IDs. Frozen giữ **cả student và base attribute head**, không chỉ riêng stereo network. [E4]

| Selected-policy metric | Joint459 | Frozen459 |
| --- | ---: | ---: |
| EPE | 8.088562 | 7.495156 |
| Stereo depth MAE | 10.436251 | 9.272147 |
| Stereo depth RMSE | 15.645196 | 15.696837 |
| Gaussian-centres Chamfer | 6.182991 | 5.634979 |
| Left PSNR | 19.754244 | 19.771357 |
| Right PSNR | 19.243779 | 19.411244 |
| Left SSIM | 0.635237 | 0.638555 |
| Right SSIM | 0.621618 | 0.627631 |
| STOP / stereo / Gaussian | 4 / 12 / 124 | 13 / 31 / 96 |
| Mean physical utility | 0.135705 | 0.156779 |
| Guard violations / unknown outcomes | 26 / 3 | 20 / 4 |

Base-only EPE: joint **8.088526583**, frozen **7.495234581**. Base right PSNR: **19.232687480** và **19.400981488**. Do đó một phần khác biệt đã có trước repair selection. Tuy vậy, joint có RMSE thấp hơn nhẹ và Gaussian risk Brier tốt hơn; không được viết “frozen tốt hơn mọi metric”.

Checkpoints được ghi ở completion report: joint SHA `ca8c637577f7952fa10fc64c75102b509198f28c29a4a8cf5a140d23a466ea52`, frozen SHA `067c5e54abbfc1628bd6d4d481e9d26209b4ec7b9b9de967729428aab87d9604`. Model/action/selector update counts: joint 459/459/1240, frozen 203/459/1236. Vì known labels khác nhau, số selector updates không hoàn toàn bằng nhau.

**Giới hạn chung C.1–C.2:** backbone ancestor đã học trên 140 diagnostic VAL images và 1,378 CAL images. `train_overlap: 0` chỉ kiểm tra training manifest hiện tại. Đây là development diagnostics có ancestry overlap, không phải unseen/final-test benchmark. RTX 5070 Ti, Torch 2.7.1+cu128 được ghi trong native reports. Không so trực tiếp các số này với 2.10 px trên datasets 8–9. [E3–E4]

### C.3. Geometry remediation và supervision controls

Matched 128→256 continuation, cùng 140 diagnostic images: [E3]

| Selected metric | Historical objective | Thêm GT metric-depth objective |
| --- | ---: | ---: |
| EPE | 8.006454 | 7.640302 |
| Depth MAE | 14.349148 | 9.616876 |
| Depth RMSE | 44.442064 | 15.029152 |
| Left PSNR | 19.703928 | 19.731085 |
| Right PSNR | 19.532353 | 19.434172 |

Geometry cải thiện lớn sau thay objective, right PSNR giảm nhẹ. Điều này cho thấy objective design quan trọng, không chứng minh joint optimisation tự thân bất khả thi. Một evaluation đầu chỉ hoàn tất 18/140 dù shell exit0; số trong bảng lấy từ **`geometry256-val-retry/result.json` hoàn tất 140/140**.

Các three-epoch controls: train datasets 1,2,3,6 (`train_1236`), 13,150 entries/epoch, 39,450 updates, validation dataset7 gồm 56 frames/55 GT-valid; 512×640. Multiscale width64/bins64; Adam 1e-3, cosine horizon10 epochs, dừng theo budget3 epochs. `schedule_complete:false` trong result metadata là dừng trước horizon10 theo protocol, không tự nó là training failure. [E6]

| Supervision | Seed | EPE | Depth MAE |
| --- | --- | ---: | ---: |
| KD + photometric | 20261006 | 3.446 | 4.383 |
| KD + photometric | 20261007 | 3.575 | 4.882 |
| KD + photometric | 20261008 | 3.623 | 4.656 |
| KD only | 20261006 | 3.648 | 4.562 |
| KD only | 20261007 | 3.752 | 5.30* |
| KD only | 20261008 | 3.695 | 4.90* |
| GT + photometric | 20261006 | 4.172 | 5.170 |
| GT + photometric | 20261007 | 3.950 | 4.840 |
| GT + photometric | 20261008 | 7.283 | 7.807 |
| Photometric only | 20261006 | 51.653 | 10275.663 |
| KD + photometric, right strip masked | 20261006 | 3.461 | 4.53* |

`*` Precision này là số đã ghi ở supplementary table; các JSON rows gốc được tham chiếu ở E6. Các giá trị khác được chép từ REPORT_C/D/G. Không coi supplementary là nguồn mạnh hơn raw rows.

Photometric-only prediction có `pred_disp_mean: 0.25` trong raw records: collapse tới configured minimum disparity, không phải mất ổn định số học với NaN đã được chứng minh. Protocol ghi smoke có gradient student/head dương. Rendering PSNR/SSIM/LPIPS và depth RMSE **không được đo bởi evaluator geometry-only của các controls này**. KD-only không học head, nên cũng không thể so rendering của nó như một matched trained renderer.

### C.4. Những joint/selector variants khác tìm thấy

Các số đều thuộc selected policy, làm tròn từ native result. Chúng là exploratory diagnostics, không phải ablation trainable centres. [E5]

| Run / artifact nhánh | Frames | EPE | Depth MAE | Depth RMSE | PSNR L / R |
| --- | ---: | ---: | ---: | ---: | --- |
| `joint-action-heads-val-aovsf_48/evaluation/result.json` | 70 | 7.85007 | 13.29427 | 39.95358 | 19.78046 / 19.35211 |
| `joint-fresh-selector-gpu1-juby_sgn/geometry203-val/result.json` | 140 | 7.49502 | 9.27209 | 15.69668 | 19.76945 / 19.40998 |
| Cùng run, `context203-val/result.json` | 140 | 7.50886 | 9.28332 | 15.71156 | 19.75227 / 19.46358 |
| `signed-gain-joint-trial-gpu0-cxydrg8v/evaluation/result.json` | 28 | 7.38174 | 9.62794 | 14.65418 | 19.70030 / 19.39383 |
| `signed-gain-long-joint-gpu0-4ri4mans/evaluation/result.json` | 140 | 7.56418 | 9.28483 | 14.57752 | 19.87323 / 19.49174 |
| `mask-token-joint-trial-gpu0-u3azczq3/evaluation/result.json` | 28 | 7.41128 | 9.64008 | 14.68873 | 19.66169 / 19.38186 |

Fresh-context pilot tăng context selector1→8, cùng parent128 và 75 additional samples. Signed-gain long dùng 2,788 model/action updates theo assessment counter. Tên `joint` không đủ để xác nhận mọi owner đã update: mask-token assessment ghi optimizer counters0; cần dùng training report/checkpoint validation cho exposure của nó, không suy từ evaluation counter. Những nhánh stationary selector refit và CAL-only giữ actor cố định khi sửa selector; chúng không phải một lần tối ưu Gaussian geometry mới. Synthetic joint/frozen probe ở `docs/joint_frozen_remediation_validation.md` cũng tồn tại, nhưng không dùng như SCARED quality evidence.

### C.5. Kiểm tra các số main GaD được người dùng cung cấp

Nguồn chính: `clean1-completion-20261107T000149+0700/results/means.json`, keys `ours_base`/`ours_refined`. Train clean10: 131,500 updates/10 epochs; full refiner1 epoch: 13,150 updates. Test datasets8–9: RGB5,907 frames, geometry5,807 frames. [E7]

| Metric | Base | Fixed-centre refined |
| --- | ---: | ---: |
| Pooled disparity EPE | 2.102800868927601 | 2.102800868927601 |
| Mean per-frame depth RMSE | 5.596403262037633 | 5.596403262037633 |
| Pooled depth MAE | 4.360801649253751 | 4.360801649253751 |
| Right PSNR dataset8 | 25.909302724194404 | 26.79711747956155 |
| Right PSNR dataset9 | 22.037104780055216 | 23.252710959266622 |
| Pooled right PSNR | 24.103325860117298 | 25.144021159363053 |
| Pooled right SSIM | 0.6889118163319677 | 0.7736782194273674 |
| Pooled right LPIPS Alex | 0.34894464333105435 | 0.24193924198764946 |
| Pooled left PSNR | 25.887906066612967 | 27.658253617268116 |

Các số tham chiếu **2.10 px**, **5.60 mm**, right gains **+0.89/+1.22 dB** khớp các artifact. “Pooled” ở đây gộp frame scores; RMSE tính trong từng frame rồi average, không phải RMSE gộp mọi pixel.

Timing artifact sau đó ghi **21.237151999912385 ms**, median qua ba invocation medians 21.20708550000927 / 21.237151999912385 / 21.325702000012825 ms. Điều kiện: RTX5070Ti, 512×640, FP32 Ours, GPU-resident input, stereo+Gaussian field+hai renders, CUDA graph; loại I/O/decode/H2D. Scope này xác nhận số21.2ms, không phải thời gian toàn pipeline đọc video. `clean1_manuscript_facts.json` cũ còn ghi latency chưa đo do failed I/O screen; record timing muộn hơn ở E7 là bằng chứng cập nhật, không xóa negative record cũ.

Không so trực tiếp main clean10 với early compact routing pilots: khác architecture, initialization, số updates, split, objective, selector, và metric aggregation/cohort. Three-epoch held-out KD control có pooled EPE2.09px, gần main2.10px; điều đó cũng ngăn cách diễn giải rằng mọi run cũ đều tệ hơn main.

### C.6. Fixed-centre controls mới hơn

`baseline-refinement-20261107T143512+0700/results/QUALITY_REPORT.json`: post hoc296 sampled test frames, 291 geometry-valid, tám keyframe clusters, một seed. Đây là cohort riêng. [E8]

| Variant | Right PSNR | Right SSIM | Right LPIPS |
| --- | ---: | ---: | ---: |
| I: identity/base | 24.092427 | 0.689273 | 0.348998 |
| MF: full refiner | 25.139930 | 0.774152 | 0.241762 |
| MS: scale-only correction | 25.108787 | 0.773632 | 0.238900 |
| L: affine/linear correction | 24.650943 | 0.741181 | 0.272597 |
| H: global-scale heuristic | 23.618180 | 0.729085 | 0.270643 |

Các variants I/MF/MS/L/H cùng EPE2.0744424302553393px và mean-frame RMSE5.065945726489693mm. H dùng `GlobalScaleCorrection` với hai log-scale constants đọc từ `heuristic/selected.json`, không phải head fine-tuning. Right PSNR thấp hơn base nhưng SSIM/LPIPS tốt hơn. Đây tiếp tục là kết quả đa metric, không phải bằng chứng centres di chuyển có hại. Code cũng hỗ trợ head-only fine-tuning, nhưng bảng này không cung cấp kết quả của arm đó. Full/scale-only comparisons từ các older eager/deployed validation cohorts cũng tồn tại; không trộn gains của chúng với main test gains.

## D. Technical explanation

### D.1. Observed evidence

1. **Base geometry thật sự thay đổi khi joint updates tiếp tục.** EPE, depth errors và Gaussian-centres Chamfer đều đổi trong C.1/C.2. Có đường gradient khả vi từ rendering về disparity. Nhưng so sánh hiện tại thay cả nhiều objective và actor states, chưa isolate riêng tác động của RGB gradient.
2. **Có gradient conflicts được đo, không đồng nhất.** `prior-failure-audit.json` ghi ở geometry128→256: RGB–geometry cosine mean−0.032737, 4/6 probes âm; KD–repair mean−0.155145, 4/6 âm. Ở large continuation: RGB–KD cosine mean+0.122478, chỉ10/41 âm. Không được khái quát thành mọi RGB/stereo gradient đều đối nghịch. Các probe thuộc model/shared gradients, không phải phép đo centre-offset riêng. [E4]
3. **Thiếu direct metric-depth objective trong historical joint code.** GT khi đó chủ yếu tạo detached counterfactual utility/risk labels, không trực tiếp supervise student/stereo repair depth. Thêm objective này cải thiện geometry lớn nhưng right PSNR giảm nhẹ. [E3]
4. **Routing/selection có lỗi chất lượng độc lập.** Joint459 có26 physical guard violations; frozen459 vẫn20. Gaussian failures chủ yếu gồm local RGB preservation; freeze student không tự giải quyết chúng. Không quy toàn bộ failure của selector cho geometry–attribute coupling. [E4]
5. **Scale implementation từng gây plateau ở SCARED-C.** Old Softplus+clamp có dead-gradient trên miền vượt scale cap; sigmoid run có render L1 tốt hơn. Nhưng 3k và10k dùng scheduler khác; đây không phải causal matched ablation. [E1]
6. **Training history từng không sạch.** Protocol_H ghi main student lịch sử được resume sau blow-up liên quan datasets4/5 với vertical stereo offset, tổng16 attempts; clean10 được tạo riêng từ untrained initialization để kiểm tra ảnh hưởng lịch sử. Không áp sự cố này cho clean10 hiện tại. [E6–E7]

### D.2. Plausible explanations

- Rendering và disparity/KD/depth objectives có thể kéo shared parameters theo các hướng khác nhau ở một số batches.
- Khi student và repairs cùng thay đổi, feature distribution, candidate regions và utility labels thay đổi theo; selector gặp nonstationary supervision.
- Attribute flexibility có thể bù lỗi projection mà không sửa metric disparity; photometric-only collapse cho thấy recipe đó không đủ ràng buộc geometry.
- Short budgets, small compact architecture, uncalibrated risk thresholds, missing direct geometry supervision và initialization khác nhau có thể đóng góp regression.

Các giải thích này phù hợp với evidence nhưng không được các run hiện có isolate hoàn toàn.

### D.3. Unverified hypotheses

- Trainable XYZ offsets gây geometry drift nặng hơn disparity-mediated updates.
- Scale/rotation/opacity instability là nguyên nhân chính trong các first-party joint pilots.
- Một loss-weight schedule hoặc detached-rendering-gradient variant sẽ giải quyết toàn bộ regression.
- Fixed centres bảo toàn rendered depth/surface, hoặc fixed-centre refinement tốt hơn mọi centre-trainable variant.

Chưa có evidence đủ để khẳng định các giả thuyết trên. Đặc biệt, fixed centres không cố định surface support/covariance: thay scale, rotation, opacity có thể thay depth compositing và visibility.

## E. Relevance to the GaD paper

| Phần paper | Có thể dùng | Giới hạn bắt buộc |
| --- | --- | --- |
| Introduction | Động cơ tránh để refinement RGB làm thay đổi stereo-derived centres; early pilots minh họa rủi ro ở recipe cụ thể | Không viết joint optimisation inherently harms metric geometry hoặc tất cả joint results đều thất bại |
| Method | Chỉ rõ centre construction canonical, stage đầu học stereo+head, stage refinement freeze model và chỉ cập nhật attributes | Phân biệt stage đầu joint training với later fixed-centre refinement; không nói pipeline chưa từng joint-train |
| Experiments | Main base/refined cùng checkpoint cho thấy RGB improvement và giữ stereo metrics; thêm matched supervision controls | Early140-image pilots có ancestry overlap phải ghi development-only; không dùng để nâng claim held-out generalization |
| Geometry–attribute decoupling contribution | Bằng chứng bảo toàn centres theo construction và hiệu quả attribute correction | Chưa có matched evidence rằng trainable-centre refinement inferior; chưa có rendered surface preservation evidence |

**Câu kết luận an toàn dựa trên artifact:**

> Gaussian attribute refinement improved rendering fidelity while retaining the stereo-derived Gaussian centres and disparity-derived metric geometry of the frozen reconstruction model.

Nếu bàn về lịch sử joint experiments:

> In bounded development pilots, some joint student–reconstruction configurations exhibited geometry regression or geometry–rendering trade-offs. These pilots do not isolate independent Gaussian-centre refinement, and matched supervision controls show that photometric training can also modestly improve stereo accuracy when combined with distillation.

Các run này chứng minh pipeline **có thể** initial-joint-train rồi fixed-centre-refine thành công; chưa chứng minh thứ tự đó tối ưu hơn mọi alternative schedule.

Các geometric quantities đã đánh giá: disparity EPE/accuracy; disparity-derived depth MAE/RMSE/AbsRel; ở early pilots còn stereo-surface Chamfer/normal consistency và Gaussian-centres Chamfer. **Rendered-depth MAE/RMSE/coverage là null trong các primary records đã đọc.** Giữ EPE không đủ chứng minh rendered surface geometry không đổi. [E2–E4, E7–E8]

## F. Missing evidence và minimal proposed ablation

**Có, vẫn cần fixed-centre versus trainable-centre ablation** nếu paper muốn khẳng định giữ centres tốt hơn cho geometry–rendering trade-off. Nếu chỉ claim khả năng cải thiện RGB đồng thời bảo toàn centres theo construction, main artifacts đã hỗ trợ; ablation bổ sung sẽ giải quyết claim so sánh mạnh hơn.

Minimal design được đề xuất, **chưa triển khai hoặc thực thi**:

1. Cùng frozen clean10 student/base head, cùng Gaussian initialization/grid/masks/colours, train order, RGB objective, optimizer budget và seed. Giữ stereo frozen trong cả hai nhánh để isolate case B versus C.
2. Fixed-centre arm: full attribute refiner hiện tại.
3. Trainable-centre arm: cùng refiner attribute và thêm zero-initialized bounded 3D offsets. Predeclare đơn vị, bounds, LR và regularizer bằng train/validation; ghi chênh parameter count. Các settings khác giữ chung.
4. Báo cáo centres displacement và centre-to-GT geometry trên support ghép cặp; render-depth/surface error và coverage theo convention/masks công bố; RGB PSNR/SSIM/LPIPS hai view; latency/memory đầy đủ. Student EPE chỉ là sanity check vì stereo frozen nên tự nó không đo centre drift.
5. Dùng ít nhất ba seeds nếu muốn claim comparative contribution vững; báo cáo per-dataset và paired keyframe-cluster uncertainty, không chọn branch/regularizer sau khi xem final test.

Nếu cần claim riêng về photometric gradient vào stereo, thêm ablation **khác**: cùng student/head initialization và KD loss, so RGB gradient về disparity bật/tắt, giữ head trainable và cùng schedule. Không thay nó bằng centre-offset comparison. Để claim benefit của hai-stage schedule, cần matched total-budget comparison giữa joint continuation và fixed-centre continuation từ cùng checkpoint.

Evidence còn thiếu: multi-seed ancestry-clean joint/frozen comparison trên current architecture; matched XYZ-centre ablation; rendered-depth geometry của main refinement; complete comparable runtime của historical joint pipelines; saturation/gradient distributions đủ để quy nguyên nhân first-party failures. Không nội suy các metric thiếu.

## G. Evidence references và phạm vi kiểm toán

### G.1. Các roots tuyệt đối

- **REPO** = `/mnt/d/workspace/huann/reliableendo-gs`
- **RUNS** = `/home/bkai/.local/share/reliableendo-gs/runs`
- **REPORTS** = `/home/bkai/.local/share/reliableendo-gs/reports`
- **EXT** = `/mnt/d/3DGS_runs/reliableendo-gs`

Các đường dẫn bên dưới ghép với root tương ứng thành đường dẫn chính xác. Với JSON, key/log identifier được ghi thay cho dải dòng; raw records không được sửa.

### E1 — SCARED-C joint Stage2 và scale bug

- REPO `outputs/stage2_scared_c_camera_local_20260919_102432/validation_metrics.csv`: rows step1000…10000; `training_log.csv`: step1…10000; `provenance.json`: protocol, split SHA, 7,749 train/422 validation; `latest.pth` và `best_validation_total_loss.pth`.
- REPO `outputs/stage2_scared_c_camera_local_scale_sigmoid_20260919_3k/validation_metrics.csv`: step3000; checkpoints/provenance cùng directory.
- REPO `outputs/stage1_scared_c_full_v1_20260915/training~/validation_metrics.csv`, `stage1_freeze.json`, checkpoint step060000.
- [Audit cũ](./../outputs/audit_stage1_60k_stage2_10k_20260920.md), phần “Metrics đọc lại”, “Lỗi scale”, “Các điểm còn thiếu”; JSON cùng stem chứa evidence kiểm tra trước đây. Đây là secondary audit, CSV/provenance ở trên là primary artifacts.
- Historical source: commit `967c339a6b5e3e9528ff61b9088e70d08fc32b48`, `src/reliable_endo_gs/training/stage2_scared_c.py`, lines264/295 optimizer toàn model, lines327–331 forward/backward; current moved path `src/reliable_endo_gs/baseline/training/stage2_scared_c.py`.
- Upstream `third_party/endo_e2e_gs/train_stage2.py:54,77–105`; `lib/network.py:39–52,67–82`; `lib/utils.py:123–129`. Depth min–max normalization upstream khiến SCARED-C run này không phải chứng cứ calibrated metric-depth fidelity.

### E2 — Joint128→1152

- RUNS `joint-larger-gpu1-4vgk9wi_/parent128-val/result.json` và `joint1152-val/result.json`: `summary.base.metrics`, `summary.selector.metrics`, `samples:140`; raw `samples.jsonl` trong các directory evaluation tương ứng.
- [Completion report](/home/bkai/.local/share/reliableendo-gs/runs/joint-larger-gpu1-4vgk9wi_/completion_report.md:3): budget, resources, checkpoint; lines13–23 comparison; lines25–37 diagnosis/limits.
- RUNS cùng root: `plan.json`, `evaluation-plan.json`, `paired-assessment.json`, `train-result.json`, `train/train.jsonl`, `train/stopped.pt`.
- Parent SHA `22f71cb96c2b744524a223bcc5406099ceee60081c660a58c4edc31080f2a425`.

### E3 — Geometry remediation / ancestry

- RUNS `joint-geometry-remediation-qayuedy8/control256-val/result.json`, `geometry256-val-retry/result.json`: `summary.selector.metrics`; raw sample logs trong cùng evaluation directories.
- RUNS cùng root: `ancestry-overlap-audit.json`, `data-audit.json`, `consolidated-evidence.json`, `control/train/result.json`, `geometry/train/result.json`.
- REPO [joint geometry audit](./joint_geometry_selector_audit_2026-10-13.md): lines18–36 objective/ancestry/data limits; lines115–145 paired results và partial-evaluation warning; lines169–195 checkpoint/source hashes.
- Configs REPO `configs/training/intervention_joint.yaml`, `configs/routing/intervention_metric_geometry_supervision.yaml`; geometry256 checkpoint SHA `2f24f2406b74a9d48f78f8cd1765bf789ca5f7220d3786bc53c0ab737bcc3fc0`.

### E4 — Joint versus frozen

- [Primary completion JSON](/home/bkai/.local/share/reliableendo-gs/runs/joint-versus-frozen-gpu0-11fuurbl/completion-report.json:1): `evaluation_scope`, `results.joint459`, `results.frozen459`, `metrics`, `base_metrics`, checkpoint identities và evidence digests.
- RUNS cùng root: `joint459-val/result.json`, `frozen459-val/result.json`, raw `samples.jsonl`; `joint459-assessment.json`, `frozen459-assessment.json`; `branch-integrity.json`, `frozen-periodic-integrity.json`, `joint-worklist.json`, `prior-failure-audit.json`.
- `prior-failure-audit.json:65`: `gradient_conflicts.geometry_joint_128_to256` và `large_joint_128_to1152`, gồm mean cosine/probe counts.
- REPO [joint/frozen documentation](./joint_frozen_student_comparison.md): lines124–146 paired protocol,158–244 results/integrity; lines83–112 freeze semantics.
- Configs: `configs/training/student_joint_routing.yaml`; `configs/routing/intervention_metric_geometry_supervision.yaml` versus `intervention_metric_geometry_frozen_student.yaml`. Reviewed source snapshot: RUNS cùng root `source-reviewed/src/`; source SHA `c48ef5b9c4944651d41b3f84fb3e19cfdd673b1bfba379f6222833b5e60d40d3`.

### E5 — Additional selector/intervention pilots

- Mỗi path trong bảng C.4 là relative to RUNS; JSON key `summary.selector.metrics`, `samples`, `checkpoint_sha256` xác định số/model được đánh giá.
- `joint-fresh-selector-gpu1-juby_sgn`: `geometry203-val/result.json`, `context203-val/result.json`; REPO `docs/joint_geometry_selector_audit_2026-10-13.md:216` ghi paired context pilot.
- `signed-gain-long-joint-gpu0-4ri4mans`: `assessment-report.json`, `plan.json`, `train-report.json`, `full-train-report.json`; checkpoint SHA `47b5d86a64188630b369546171b4b01e0439ae61b33ef4e6246104e7d96ddf3c`.
- `mask-token-joint-trial-gpu0-u3azczq3`: `assessment-report.json`, `train-report.json`, `checkpoint-validation.json`, `plan.json`; checkpoint SHA `d4c34e7181c5d9be65956121a061b05b06ebe518fc7b860f4ff8b8563394a684`.
- REPO `docs/selector_joint_optimization.md:55–108` ghi các 168-update negative selector pilots; chúng kiểm tra selector optimization, không phải B-versus-C.
- REPO `docs/joint_frozen_remediation_validation.md:75–112`: analytic synthetic probe; không dùng như native dataset result.

### E6 — Supervision controls / photometric collapse

- REPORTS `insight-ablations-20261101/plan/PROTOCOL.txt:11–23`, `PROTOCOL_C.txt:3–24`, `PROTOCOL_D.txt`, `PROTOCOL_G.txt`; `plan/student_rec_only.yaml`, `student_kd_only.yaml` và seed variants.
- [Photometric-only report](/home/bkai/.local/share/reliableendo-gs/reports/insight-ablations-20261101/results/REPORT_C.txt:1); cùng directory `REPORT.txt:4–14`, `REPORT_D.txt`, `REPORT_G.txt`, `REPORT_E.txt`.
- Primary per-frame JSON: REPORTS `insight-ablations-20261101/results/job3/reconly_ep3.json`, `kdonly_ep3.json`, `kdseed2_ep3.json`, `kdseed3_ep3.json`, `kdonlyseed2_ep3.json`, `kdonlyseed3_ep3.json`, `gtseed2_ep3.json`, `gtseed3_ep3.json`, `kdvisible_ep3.json`; keys `info.cursor`, `info.sha256`, `rows[].epe`, `depth_mae_mm`, `pred_disp_mean`.
- Matched seed1 KD/GT controls được protocol trỏ đến REPORTS `kd-attribution-20261031/results/job3/kd_matched_ep3.json` và `gt_ep3.json`; các số seed1 được đọc từ generated REPORT_C/D/G.
- EXT `insight-ablations-20261102-reconly/reconly/attempt-00/result.json`, `train.jsonl`, `stopped.pt`; checkpoint SHA `e9c994dafe71105272f081ef2945f8fcc6de1fc175432f2455d6d083215fc204`; cursor39,450, epoch3, `schedule_complete:false`.
- EXT `insight-ablations-20261101/kdonly/attempt-00/result.json`, `train.jsonl`, `stopped.pt`; same budget; SHA `9cce469cd9b19a2112af8b3885cdd76df36bcab6247e3c215b1170692eb5ba84`.
- REPO `paper_overleaf/supp.tex:729–765` là secondary rendering của seed table, không phải nguồn primary.

### E7 — Main clean10 / fixed-centre refinement / timing

- [Main saved means](/mnt/d/3DGS_runs/reliableendo-gs/clean1-completion-20261107T000149+0700/results/means.json): keys `ours_base`, `ours_refined`, metric→`dataset_8`/`dataset_9`/`pooled`, mỗi record chứa scalar và frame count. `comparison_tables.md` là bảng derived để đọc.
- EXT `clean10-20261103/kd/attempt-00/result.json`: epoch10,131,500 updates, `schedule_complete:true`. `FINAL.pt` SHA `db7413e8d456177ee55eb7e3cb75580b5cfd40c443403359f8173fb21a6131b6`.
- EXT `clean10-migration-20261106T143532+0700/refiner/spec.json`, `summary.json`: inputs/bounds/seed/LR,13,150 steps,1,671 parameters, `frozen_unchanged:true`; `refiner.pt` SHA `aa6736393b15cad40d59dc834ca18e7f2a19817979d972bd3a541610e34fc9a6`.
- REPORTS `clean10-20261103/plan/PROTOCOL_H.txt:1–8`: clean initialization và lịch sử run cũ; giữ bản superseded để phân biệt versions.
- [Matched timing result](/mnt/d/3DGS_runs/reliableendo-gs/matched-optimized-latency-20261107T004544+0700/results/result.json): `scope`, `precision`, `models`, `ours_ms`, `ours_median_ms`; raw `raw/ours_run1.json`…`ours_run3.json` ghi hardware/timing details.
- REPO `docs/PAPER_CODE_RESULTS_CHECK_2026-11-11.md:25–34` giải thích frame aggregation và rendered-surface limits; đây là secondary verification đã có trước audit này.

### E8 — Fixed-centre alternative correction controls

- [Quality report](/mnt/d/3DGS_runs/reliableendo-gs/baseline-refinement-20261107T143512+0700/results/QUALITY_REPORT.json): `means.I/MF/MS/L/H.pooled`, `paired`, `limits`.
- REPO `src/reliable_endo_gs/evaluation/baseline_refinement_study.py:50,90–115`: mapping variants; `training/field_refinement.py:105–117` head/refiner optimizer ownership.
- Historical component report: REPORTS `insight-ablations-20261101/results/REPORT.txt:16–26`; không thay thế clean/cohort khác bằng số historical này.

### E9 — Canonical implementation / historical source

- REPO `src/reliable_endo_gs/reconstruction/student.py:297–305,327–350`: trainable disparity vào reconstruction.
- REPO `src/reliable_endo_gs/reconstruction/model.py:614–680`: geometry, head, composition, render.
- REPO `src/reliable_endo_gs/gaussians/composition.py:284–306`: centres lấy từ prior; attributes được compose riêng.
- REPO `src/reliable_endo_gs/training/student_learning.py:200–224,544–613`: optimizer ownership, detach selector, RGB/KD/repair loss.
- REPO `src/reliable_endo_gs/training/balanced_learning.py:108–150,475–521`: warp/KD/geometry và dynamic weights.
- REPO `src/reliable_endo_gs/training/field_refinement.py:105–117,197–228`: freeze, no-grad, trainable ownership.
- REPO `src/reliable_endo_gs/actions/refinement.py:84–103,164–166`: giữ means và disparity; main colour residual disabled ở spec E7.
- REPO `docs/student_joint_learning.md:150–180`: compact parameter counts và simple loss definitions.
- Git `b8e61be`: thêm student/joint learning; historical `reconstruction/student.py:191–217` đã giữ trainable prediction thay frozen teacher detach. `bc16ed7`: masked joint pipeline. `96ffbc8`: deployed refinement. `9a96d31`: zero photometric/restricted outputs. Git history search các identifiers `trainable_centers|center_offset|position_delta` trong first-party src/config/scripts không tìm thấy commit tương ứng; đây không phải chứng minh tuyệt đối về mọi branch/artifact từng tồn tại.

### G.2. Coverage và những việc không làm

Đã tìm source/config/scripts/tests/docs, ignored local outputs/paper folders, archived reports, các Git commits có liên quan và external run roots theo provenance. Đã kiểm tra nội dung text trong ba archive `paper_support/_old/paper_overleaf.zip`, `paper_overleaf.zip`, `paper_support/updates/20261102/overleaf_upload.zip`; không thấy additional direct-centre experiment trong các phần text match. Không phát hiện notebook first-party liên quan trong file inventory đã tìm. Những launch-only directories, synthetic pytest artifacts và duplicate assessments không được tính thành independent scientific experiments.

**Implemented:** chỉ báo cáo Markdown này, theo yêu cầu bổ sung. **Verified:** đọc primary scalars/config/source, phân biệt cohort/stage/gradient flow, đối chiếu recorded results và status. **Not tested:** không chạy tests, không load/rerender checkpoint, không rehash toàn bộ artifact lớn, không independently reproduce metrics. **Unavailable:** direct B-versus-C result và rendered-surface preservation evidence. **Planned only:** các ablations ở F. Không có model/config/result/paper nào được sửa bởi audit.
