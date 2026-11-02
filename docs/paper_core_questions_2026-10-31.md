# Năm câu hỏi trọng tâm của bài báo (What / Why / How / Results / Contributions)

Trạng thái: **bản nháp nội bộ, 2026-10-31**, chưa commit. Mọi con số lấy từ các lượt chạy đã ghi lại (nguồn ở cuối file).
Nhãn bằng chứng dùng xuyên suốt: **dev** = 56 frame benchmark val_7 (một chuỗi, 4 keyframe, thường một seed);
**confirmatory** = lượt chạy đã đăng ký trước trên sealed-64 và final_test (dataset 8, 9); **thăm dò** = xem sau khi đã biết kết quả, không đổi được verdict.
Theo `AGENTS.md`: chưa có bằng chứng thì viết "giả thuyết", không viết như kết luận.

---

## 1. What? Vấn đề là gì

**Bài toán.** Từ **một cặp ảnh stereo nội soi đã rectify** (kèm calibration), dựng ngay trong một lượt feed-forward một **trường Gaussian 3D** ở hệ tọa độ camera trái
(theo thang mét nhờ stereo), rồi render lại ảnh trái và phải và lấy ra disparity/depth. Mỗi frame được xử lý độc lập: không tối ưu theo từng cảnh, không bộ nhớ hay mô hình thời gian.
Ảnh render là ảnh ở hai camera đầu vào, nên đây là kiểm tra **nhất quán giữa hai view**, chưa phải tổng hợp góc nhìn mới.

**Vấn đề cụ thể (hai tầng).**

1. **Tổng quát hóa giữa các chuỗi phẫu thuật.** Phương pháp feed-forward cùng task đã có (Endo-E2E-GS) huấn luyện bằng disparity ground truth. Khi train lại theo cấu hình matched của ta và
   thử trên chuỗi chưa thấy (val_7), nó cho EPE **15,47 px**, trong khi student của ta đạt **3,62 px** và teacher 3,19 px. Thử nghiệm dịch disparity một hằng số
   (±24 px) cho thấy dự đoán của baseline gần như không đổi (slope **−0,01** trên train, **0,01** trên val_7; student 0,99 và 1,01; teacher 1,00 và 1,00): nó thoái hóa thành một prior gần hằng
   thay vì dùng tín hiệu stereo. Checkpoint SCARED-C giai đoạn 1 (do dự án huấn luyện bằng mã gốc trên dataset 1 đến 3) cũng sai 11,3 px trên val_7.
2. **Chất lượng nhất quán hai view còn dư.** Trường Gaussian dựng từ prior hình học có kích thước và độ trong suốt theo quy tắc viết tay, nên ảnh render ở view phải còn sai lệch.

**Giới hạn cần nói thẳng.** Baseline là **bản train lại matched**; sự tương đương với bài gốc ("native parity") **chưa được kiểm chứng**. Kết quả trên mới là dev (val_7), chưa chạy trên final_test.

---

## 2. Why? Tại sao cần giải quyết

- **Tín hiệu stereo phải được dùng thật.** Một mô hình bỏ qua stereo và trả về prior gần hằng sẽ cho hình học sai ở cảnh mới, mà ngoài phòng mổ mỗi ca là một "cảnh mới". Đây là lỗi nguy hiểm vì ảnh render vẫn trông hợp lý.
- **Disparity ground truth khan hiếm trong nội soi.** Nhãn thường từ thiết bị đặc biệt hoặc giả nhãn, khó mở rộng. Cần cách huấn luyện **không cần GT disparity** nhưng vẫn tổng quát hóa.
- **Phương pháp theo từng cảnh quá chậm cho dùng trong mổ** (phải tối ưu lại cho mỗi cảnh). Hướng feed-forward cho kết quả ngay trên từng frame, nhưng trong tìm kiếm có hạn của chúng tôi chỉ thấy **một** phương pháp feed-forward Gaussian từ cặp stereo nội soi (Endo-E2E-GS),
  nên chưa có cơ sở so sánh vững.
- **Tốc độ và chất lượng.** Dùng teacher nền tảng chậm ở lúc chạy là không thực tế; cần một student nhanh giữ được phần lớn độ chính xác, kèm một bước hiệu chỉnh rất nhẹ để lấy lại chất lượng render.

(Phần động cơ này dựa trên tài liệu chung và chưa được đo trong dự án; các câu về nhu cầu lâm sàng cần trích dẫn khi viết bài.)

---

## 3. How? Phương pháp

Ba khối, đều chạy trên một frame:

**(a) Student stereo chưng cất từ teacher, không cần GT.** Student đa tỉ lệ (multiscale) học disparity của teacher FoundationStereo đóng băng bằng loss SmoothL1 (`kd_weight` 1), cùng loss tái dựng
Charbonnier trên cả hai view (`reconstruction_weight` 1). Tập train_1236 (SCARED dataset 1, 2, 3, 6); dataset 4 và 5 bị loại vì lệch dọc 17–22 px không giải thích được bằng calibration; dataset 7 là validation; dataset 8 và 9 là final_test niêm phong.
Lúc suy luận không cần teacher.

**(b) Dựng trường Gaussian bằng công thức chuẩn.** Disparity → depth → tâm 3D → một Gaussian cho mỗi ô của lưới lấy mẫu cố định trên view trái; một đầu thuộc tính dự đoán residual có biên; màu **chỉ lấy từ ảnh trái**
(cả hai ảnh trái và phải là đầu vào của student stereo; ảnh phải còn là mục tiêu của loss ảnh; Gaussian chỉ nhận màu từ ảnh trái). Mỗi công thức khoa học có đúng một nơi cài đặt.

**(c) Refiner siêu nhẹ (L6, 1.671 tham số).** MLP 11→32→32→7 trên 11 kênh ngữ cảnh, dự đoán residual cho xoay, log-scale và logit opacity của từng Gaussian, nằm trong biên đặt tay (×2) và ghép qua cùng hàm
`compose_gaussian_field`. Huấn luyện bằng **loss quang trắc Charbonnier trung bình hai view** (so ảnh render với chính ảnh đầu vào), student đóng băng và được kiểm tra băm trước/sau. Khi residual bằng 0 thì đúng bằng STOP.

**Triển khai.** Bố cục lưới tĩnh và một CUDA graph từ stereo đến trường đã refine (cấu hình B1, chính xác từng bit với đường tham chiếu); C2 thêm `torch.compile` cho 4 kernel thuần (xấp xỉ mức làm tròn).

**Giao thức đánh giá.** Mọi ngưỡng thành công được viết và băm **trước** khi chạm dữ liệu niêm phong (`SUCCESS_BAR.txt`, sha 54938b46…): P1–P4 cho mức tăng, N1–N4 cho không-kém-hơn
(trong đó N3: sai số ở 80 cột cuối của view phải ≤ 1,05 lần STOP). Cấu hình đóng băng B1 là chính, C2 là phụ.

**Phần thăm dò sau verdict (không phải phương pháp chính).** Guard dải mép: giữ ảnh STOP ở dải cột mà view trái không thấy; chỉ phát triển trên dev, mặc định tắt (xem mục 4.5).

---

## 4. Results? Kết quả đạt được, có chứng minh được hiệu quả không?

### 4.1 Trả lời thẳng

**Chưa chứng minh đầy đủ.** Bằng chứng hiện có ủng hộ một số ý và còn thiếu ba mảnh quan trọng:

1. Claim chính về refiner đã được đánh giá confirmatory và **không đạt** theo thanh đã đăng ký, vì trượt đúng một điều kiện (N3 trên dataset 9).
2. Chưa có so sánh nào với baseline trên **final_test** (chỉ có dev); chưa đo hình học (EPE, depth MAE) trên final_test; chưa đo FPS/latency cho lượt cuối.
3. Claim "chưng cất tốt hơn học bằng GT" **chưa đạt ngưỡng đăng ký** (inconclusive).

### 4.2 Bằng chứng theo từng ý

| Ý | Bằng chứng | Mức | Kết luận |
|---|---|---|---|
| Baseline thoái hóa giữa chuỗi | EPE 15,47 px (baseline) so với 3,62 (student) và 3,19 (teacher); slope −0,01/0,01 so với 0,99/1,01 | dev (val_7) | Ủng hộ trên một chuỗi. Chưa có final_test, chưa có dữ liệu ngoài |
| Nguyên nhân là giám sát bằng GT | Student cùng kiến trúc học bằng GT vẫn đạt 4,17 px | dev | **Không ủng hộ** cách giải thích "GT là lỗi" |
| Student giữ độ chính xác của teacher | EPE 3,62 so với 3,19 px; depth MAE 4,37 so với 3,97 mm | dev, 55 frame | Mô tả |
| Chưng cất tốt hơn GT (khớp kiến trúc, lịch, seed) | EPE 3,45 so với 4,17 px, depth MAE 4,38 so với 5,17 mm, KD thắng 46/55 frame; r = 1,211 (ngưỡng 1,25) | dev, một seed, 4 keyframe | **Mixed/inconclusive** theo luật đăng ký. Chỉ nói "EPE thấp hơn trong so sánh matched" |
| Refiner tăng chất lượng view phải | +0,99 dB (KD 10 epoch), +1,16 dB (GT-only), +0,92 dB (KD matched), dương trên 96–98% frame | dev | Chung cho 3 student đã thử |
| Refiner ổn định theo seed | +0,994 / +0,951 / +1,126 dB (range 0,175 dB ≤ 0,3) | dev, một student | Ổn định |
| Mức tăng đến từ đâu | Tắt residual scale: +0,99 → −0,19 dB; tắt xoay hoặc opacity: không đổi | dev | Mức tăng đến từ **residual scale** |
| Bền với nhiễu đầu vào | 7 điều kiện, +0,57 đến +1,11 dB, dương ≥ 95% frame | dev | Mô tả |

### 4.3 Kết quả confirmatory (L6 đóng băng, B1)

| Tập | Số frame | Right-PSNR gain (CI 95%) | Frame dương | Right SSIM | Right LPIPS | N3 (≤ 1,05) |
|---|---|---|---|---|---|---|
| Sealed 64 (dataset 7) | 64 | +1,154 [0,996; 1,312] | 96,9% | – | – | – |
| final_test dataset 8 | 3.152 | +0,599 [0,581; 0,616] | 90,3% | +0,030 | −0,033 | 0,986 ✓ |
| final_test dataset 9 | 2.755 | +1,328 [1,307; 1,349] | 99,5% | +0,063 | −0,062 | **1,054 ✗** |

Điểm tuyệt đối trung bình (STOP → + refiner): dataset 8, left PSNR 28,03 → 29,91, right 26,49 → 27,09; dataset 9, left 23,40 → 25,50, right 22,53 → 23,86.
Các điều kiện P1–P4, N1, N2, N4 đều đạt trên cả hai dataset; không có frame nào bị lỗ. Keyframe_0 của dataset 8 chỉ tăng +0,12 dB.
**Verdict đã đăng ký: "not supported"**, chỉ vì N3 trên dataset 9. Cấu hình C2 cho kết quả giống B1 (trong 0,001 dB, N3 1,0542), nên việc chọn B1 không làm đổi kết luận.
Hệ quả theo thanh đã viết: claim sửa lỗi chỉ được nêu như bằng chứng phát triển, kèm số đo thật và N3 được công khai; **thanh không được sửa**.

### 4.4 Vì sao trượt N3 (thăm dò, sau khi biết verdict)

- Refiner **chỉ tệ hơn STOP ở khoảng một disparity cuối của view phải** (tỉ lệ MSE 1,15 ở mép, về 1,0 khoảng 40 cột trong mép trên dataset 9); từ khoảng 45 cột trở vào thì tốt hơn (0,80–0,93). View trái tốt hơn ở mọi băng cột.
- Trên dev, điểm đảo dấu dịch theo disparity mép (≈39 / ≈54 / ≥100 cột cho disparity 42 / 61 / 83 px). Phù hợp với giả thuyết các cột view phải không có đối ứng ở view trái không được sửa tốt, vì refiner chỉ đổi dạng Gaussian đã có
  (không có đầu ra màu). **Giả thuyết này chưa được kiểm chứng trực tiếp.**
- Ở dev, dải này chiếm khoảng 41% MSE view phải của STOP, nên còn dư địa lớn nhưng không chắc lấy lại được.

### 4.5 Guard dải mép (chỉ dev, chưa phải kết quả)

Giữ ảnh STOP ở dải cột này. Trên 56 frame dev: N3 1,0278 → 0,9949, right-PSNR gain +0,996 → +1,102 dB; strip cố định 60 cột trượt các nhóm disparity trung bình/cao. Điều kiện (a) đạt gần như theo cấu trúc, phần có thông tin là các band ngoài dải vẫn ≤ 1,0.
Hạn chế: độ rộng dải có đuôi nặng (35–221 px), một frame tệ đi (1,17 → 0,83 dB), chưa kiểm tra bằng mắt đường nối, chi phí thêm một lần rasterize chưa đo. Đây là **"repair withheld where the right view has no left-view counterpart"**, không phải "đã sửa lỗi mép".

### 4.6 Tốc độ

B1 55,5 FPS và C2 61,4 FPS ở 512×640 (trung vị 3 lần chạy, nghiên cứu FPS riêng ngày 30/10, refiner tốn 0,81 ms). Chưa đo lại cho lượt cuối (GPU dùng chung) và **chưa đo cho đường có guard**; baseline chưa được xử lý bằng CUDA graph nên **không có câu so sánh tốc độ**.

### 4.7 Cần gì để đóng khoảng trống

| Còn thiếu | Việc phải làm |
|---|---|
| So sánh với baseline trên final_test | Chốt danh sách baseline vào `COMPARISON_PLAN.md` và băm; chạy đúng một lượt đã đăng ký (cần bạn đồng ý, vì đây là lần mở final_test thứ ba) |
| Hình học trên final_test | Có trong lượt trên (EPE, depth MAE) |
| KD so với GT | Thêm dữ liệu ngoài hoặc thêm chuỗi, đăng ký lại ngưỡng; thêm seed không đủ vì chỉ có 4 keyframe độc lập |
| Chi phí | Đo latency/FPS trên GPU rảnh, kể cả đường có guard |
| Dữ liệu ngoài | EndoNeRF (cần chấp nhận điều khoản tải) hoặc SERV-CT |

---

## 5. Contributions? Điểm mới và đóng góp

**Định vị.** Gaussian splatting feed-forward từ cặp stereo đã có ngoài y khoa (GPS-Gaussian, Stereo-GS, StereoGaussians) và một đại diện nội soi gần đây (Endo-E2E-GS).
Chưng cất mô hình stereo nền tảng đã có cho ảnh tự nhiên. Tinh chỉnh residual theo từng Gaussian đã có trong feed-forward GS. **Bài không thể claim các ý này.**
Phần có thể đứng được là một **chẩn đoán thực nghiệm, một công thức huấn luyện không cần GT và một hiệu chỉnh scale rất nhỏ, đánh giá giữa các chuỗi.**

| ID | Đóng góp | Điểm mới | Trạng thái bằng chứng | Cách viết cho phép hôm nay |
|---|---|---|---|---|
| **C1** | **Chẩn đoán**: mô hình feed-forward học bằng GT thoái hóa thành prior gần hằng khi chuỗi test vượt dải disparity train; kèm **giao thức thử nghiệm dịch disparity** (crop bù, thay đổi hằng số, giữ nguyên thang) dùng lại được | Khung phát hiện lỗi có thể tái sử dụng; lỗi này không lộ ra qua PSNR | dev (val_7), hai checkpoint baseline đều sai | "trên chuỗi chưa thấy, baseline tái huấn luyện cho EPE 15,5 px, slope 0,0" (cần final_test và dữ liệu ngoài) |
| **C2** | **Công thức không cần GT**: chưng cất teacher stereo nền tảng, student giữ phần lớn độ chính xác (3,62 so với 3,19 px) và EPE thấp hơn student học bằng GT cùng kiến trúc (3,45 so với 4,17 px) | Đối chứng khớp kiến trúc/lịch/seed trong nội soi | dev, một seed; **r = 1,211 < 1,25** | "EPE thấp hơn trong so sánh matched (một seed, một chuỗi, dưới ngưỡng đã đăng ký)"; **không** viết "tốt hơn", "có ý nghĩa" hay "cải thiện tổng quát hóa" |
| **C3** | **Hiệu chỉnh scale 1,7 nghìn tham số**, một lượt, không phản hồi; tăng nhất quán view phải; mức tăng đến từ residual scale | Thành phần nhỏ, ablation cho thấy cơ chế | dev ổn định (3 student, 3 seed); confirmatory: +0,60 dB và +1,33 dB nhưng **verdict not supported** | "tăng right-PSNR +0,6 dB (dataset 8) và +1,3 dB (dataset 9); không đạt điều kiện không-kém-hơn N3 trên dataset 9"; không viết "đã được xác nhận" |
| **C4** | **Tốc độ** | Chỉ có ý nghĩa nếu baseline được tối ưu tương đương | Đã đo riêng 55,5/61,4 FPS; chưa so được | Chỉ nêu FPS của ta, không so với baseline |
| **C5** | **Thực hành đánh giá**: ngưỡng thành công băm trước khi chạm dữ liệu niêm phong; kết quả âm được công bố nguyên trạng, kèm phân tích nguyên nhân (dải mép) | Minh bạch, tái lập được | Hoàn tất cho lượt STOP so với refined | Có thể nêu như phương pháp luận, không như đóng góp kỹ thuật |
| (phụ) | Phát hiện âm: selector theo vùng không giúp được (Gate A NO-GO, oracle chỉ +0,004 dB) → thay bằng refiner toàn khung | Kết quả âm có ích | dev | Mục kết quả âm |

**Rủi ro phản biện (đã biết).**
- "Chỉ là hiệu chỉnh một quy tắc kích thước Gaussian": cần chứng minh trên backbone khác hoặc dữ liệu khác.
- "Baseline chưa tái lập trung thực": đã công khai; checkpoint SCARED-C do dự án huấn luyện cũng sai theo cách tương tự.
- "Chỉ một bộ dữ liệu": cần dữ liệu ngoài.
- "View phải nằm trong loss": nói rõ; claim là nhất quán giữa hai view trên frame chưa thấy, với màu chỉ lấy từ ảnh trái.
- "Bài ứng dụng": sức nặng kỹ thuật phải đến từ C1 (giao thức và phát hiện) và C2 (đối chứng matched); C3 một mình không đủ cho CVPR. C2 hiện chưa đạt ngưỡng, nên trọng tâm sẽ rơi vào C1 và C3.

---

## Nguồn số liệu (để truy vết)

- Dev, KD attribution, refiner: `~/.local/share/reliableendo-gs/reports/kd-attribution-20261031/results/REPORT.txt`, `results/job3/*.json`.
- Confirmatory: `~/.local/share/reliableendo-gs/reports/confirmatory-20261031/` (`SUCCESS_BAR.txt`, `FROZEN_CHOICE.txt`, `results/{B1,C2}_{s1,s2}/report.json`, `rows.jsonl`).
- Phân tích dải mép: `confirmatory-20261031/analysis/` (`EXPLORATORY_NOTES.txt`, `EDGE_GUARD_DEV_SPEC.txt`, `EDGE_GUARD_DEV_RESULT.txt`).
- Ledger claim: `docs/paper_contributions_draft_2026-10-31.md`. Tốc độ: `docs/fps_bc_status_2026-10-30.txt`.
- Điểm tuyệt đối trung bình ở 4.3 tính từ `results/B1_s2/rows.jsonl`.
