# Các thí nghiệm còn cần làm cho bài báo (2026-10-31)

Trạng thái: **bản đánh giá khoảng trống, chưa chạy gì thêm.** Căn cứ: các lượt chạy đã ghi (xem `docs/paper_core_questions_2026-10-31.md`), báo cáo của Codex
(`docs/baseline_feasibility_addendum_2026-10-31.md`, `docs/baseline_finetune_plan_2026-10-31.md`) và hướng dẫn thí nghiệm của skill `nature-writing`
(`references/experiments.md`: so sánh với baseline mạnh, ablation cho từng thiết kế, kiểm tra ngoài phân phối và nêu giới hạn).
Ước lượng thời gian bên dưới là **ước lượng**, chưa đo.

## 1. Đã có và không cần chạy lại

| Nội dung | Có gì | Mức |
|---|---|---|
| Hiệu chỉnh của refiner trên chuỗi chưa thấy | Right-PSNR +0,60 dB (sequence 8), +1,33 dB (sequence 9), 5.907 frame, CI 95% | held-out test |
| Ablation refiner | Bậc thang L0 (hằng số), L1 (tuyến tính), L2 đến L4 (dung lượng), L5 (màu), L6, đầu thuộc tính tinh chỉnh (H); tắt từng nhóm residual (A3); bền nhiễu (A4) | validation |
| Refiner chung cho 3 student và ổn định theo seed | 3 student: +0,92 / +1,16 / +0,99 dB; 3 seed trên một student: +0,951 đến +1,126 dB | validation |
| Chẩn đoán baseline + thử nghiệm dịch disparity | Slope −0,01/0,01 so với 0,99/1,01 | validation (một chuỗi) |
| KD so với GT, khớp kiến trúc | EPE 3,45 so với 4,17 px | validation, một seed |
| Phân tích theo cột và guard dải mép | Xem mục 4.4 và 4.5 của báo cáo năm câu hỏi | validation + thăm dò |

## 2. Việc phải làm (ưu tiên P0: thiếu thì bài không đứng được)

| ID | Thí nghiệm | Vì sao | Cần gì | Chi phí ước lượng |
|---|---|---|---|---|
| **E1** | **So sánh chính trên test sequences 8 và 9** (render + hình học): student cơ sở và + refiner; student GT-only và KD matched (kèm refiner); Endo-E2E-GS (2 checkpoint SCARED-C do dự án huấn luyện + bản train lại matched); teacher (chỉ hình học) | Không có bảng này thì không có câu "so với baseline". Hiện chỉ có phần STOP so với refiner | Chốt danh sách vào `COMPARISON_PLAN.md` và băm; **bạn đồng ý** vì đây là lần mở dữ liệu test thứ ba. Chỉ suy luận, checkpoint đã có (refiner của KD matched đã xong) | Vài giờ GPU, 5.907 frame mỗi mô hình |
| **E2** | **Hai baseline render tinh chỉnh theo protocol của ta**: MVSplat, DepthSplat-small (train_1236, 39.450 mẫu, loss quang trắc hai view, checkpoint cuối) | Bảng render chỉ có Endo-E2E-GS là quá mỏng | Codex đã chốt near/far từ train: **0,0290 m và 0,1574 m** (băm trước khi đụng val/test), nên câu hỏi near/far (a) hay (b) **đã đóng**. Còn chặn: (i) **GPU cap 20% ≈ 3,2 GiB làm cả 4 mô hình OOM**, cần bạn cho nâng cap; (ii) thứ tự mẫu: replay đúng thứ tự manifest hay shuffle cố định có công bố; (iii) kiểm tra chiếu đúng với tâm ảnh lệch | Mỗi mô hình tối đa khoảng 2 ngày kèm cổng kiểm tra; mặc định 2 mô hình |
| **E3** | **Baseline hình học**: HK-Shi dual-branch CNN, MSDESIS (EPE, depth MAE) | Trục hình học cần so với phương pháp stereo chuyên dụng ngoài teacher | Lấy weights, kiểm tra checkpoint có train trên SCARED 8/9 không | 1–2 ngày mỗi cái; có thể cắt còn một |
| **E4** | **Thử nghiệm dịch disparity trên test 8, 9** cho mọi mô hình có disparity | C1 hiện chỉ dựa trên một chuỗi validation | Dùng lại giao thức đã có | Thấp (suy luận) |
| **E5** | **Dữ liệu ngoài** (EndoNeRF có ảnh phải trong thư mục công khai, hoặc SERV-CT 16 cặp): zero-shot cho student, refiner, baseline | Phản biện "chỉ một bộ dữ liệu"; cũng là dữ liệu sạch duy nhất để đánh giá guard dải mép | **Bạn phải chấp nhận điều khoản tải EndoNeRF**; calibration và baseline của EndoNeRF không được ghi, phải ước lượng hoặc tìm nguồn và nêu rõ giả định | Chưa rõ; kiểm tra khả thi trước (nửa ngày) |
| **E6** | **Đo tốc độ trên GPU rảnh**: student, + refiner (B1/C2), đường có guard; Endo-E2E-GS; các baseline tinh chỉnh; kèm phân rã từng khâu | Dòng chi phí chưa đo; không được trích 55,5 FPS cho đường có guard | GPU phải rảnh: xếp **sau** các lượt train, ghi phần cứng và load average | Vài giờ |

## 3. Nên làm (P1)

| ID | Thí nghiệm | Ghi chú |
|---|---|---|
| E7 | Bảng ablation duy nhất cho các đóng góp (từ bậc thang L0 đến L6 và H) | Chủ yếu là viết lại số liệu đã có (validation, 48 frame). View phải so với STOP: hằng số 14 tham số −0,02 dB; tuyến tính 98 tham số +0,90 dB; MLP32 +1,03 dB; L6 +1,12 dB; tinh chỉnh đầu thuộc tính +0,68 dB. Tức một hàm tuyến tính đơn giản đã đạt phần lớn mức tăng, và hiệu chỉnh toàn cục không đủ |
| E8 | Ablation chưng cất: student chỉ dùng loss quang trắc (`kd_weight = 0`) cùng cấu hình matched | Tách đóng góp của teacher khỏi kiến trúc và loss quang trắc. Cấu hình `student_joint_no_kd.yaml` có sẵn nhưng là bản cũ, cần viết lại cho khớp. Khoảng 9–10 giờ GPU |
| E9 | Hình định tính: luật chọn frame cố định (frame trung vị, frame tệ nhất, frame dải mép), render hai view, bản đồ sai số, disparity | Phải có cả trường hợp lỗi. Dùng skill `nature-figure` |
| E10 | Thống kê: bootstrap theo keyframe bên cạnh bootstrap theo frame; bảng theo keyframe | Không tốn tính toán. Mỗi dataset chỉ có 4 keyframe độc lập |

## 4. Tùy chọn (P2, chỉ khi dư thời gian)

- Đánh giá guard dải mép trên dữ liệu ngoài, kèm đo độ trễ của lần rasterize thêm.
- Refiner cắm vào backbone khác (ví dụ chạy trên đầu ra của Endo-E2E-GS) để trả lời "chỉ hiệu chỉnh một quy tắc heuristic".
- Thử nghiệm góc nhìn mới bằng tư thế kinematic/frame lân cận của SCARED. Chưa kiểm tra dữ liệu có tư thế hay không; nếu không thì để Future Work.
- Thêm seed cho student: chỉ nâng sức mạnh thống kê nếu có thêm chuỗi, vì mỗi dataset chỉ có 4 keyframe.

## 5. Không cần làm
Tìm thêm baseline (đã khảo sát hai vòng); selector hoặc router; phương pháp đơn nhãn hoặc video (Endo3R và các mô hình depth khác); phương pháp tối ưu theo từng cảnh (chỉ trích dẫn); thêm dung lượng refiner (bậc thang: MLP128 chỉ hơn MLP32 khoảng +0,05 dB ở view phải);
độ phân giải 1280×1024 (lưới bị cắt ở 100.000 Gaussian, nêu như hạn chế).

## 6. Quyết định cần bạn đưa ra

1. **Đồng ý một lượt so sánh duy nhất trên test 8 và 9** với danh sách baseline đã chốt (E1, E3, E4).
2. **Nâng GPU cap** cho tinh chỉnh baseline: cho phép một tiến trình dùng gần hết 16 GB của một GPU (hay cả hai)? Hiện cap 20% làm cả 4 mô hình OOM.
3. **Thứ tự mẫu khi tinh chỉnh baseline**: replay thứ tự manifest (khớp với các student) hay shuffle cố định có công bố là không khớp.
4. **Chấp nhận điều khoản EndoNeRF** (E5), hoặc chọn SERV-CT nếu không.
5. **Số baseline cuối cùng**: mặc định 2 render + 1–2 hình học.

## 7. Lịch tham khảo (ước lượng, tuần tự một GPU cho phần train)

- Ngày 1: chốt các quyết định trên; ghi danh sách vào `COMPARISON_PLAN.md`; kiểm tra khả thi dữ liệu ngoài.
- Ngày 2–7: tinh chỉnh MVSplat rồi DepthSplat (mỗi mô hình khoảng 2 ngày, có thể ít hơn nếu bước train nhanh); chạy E3 trên GPU còn lại nếu bạn cho phép hai tiến trình.
- Ngày 8–10: lượt so sánh E1 và E4 trên test; E5 nếu khả thi; E8 nếu còn GPU.
- Ngày 10–11: đo tốc độ E6 trên GPU rảnh.
- Ngày 11–14: hình, bảng sinh tự động, viết bài.

Nếu chỉ một baseline render qua được cổng kiểm tra thì bảng render có hai dòng cộng các đối chứng nội bộ; ghi rõ lý do các slot còn lại bị bỏ.
