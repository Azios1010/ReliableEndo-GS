# Thí nghiệm CVPR cần và mức độ phủ (2026-11-01)

Nguồn tiêu chí (bộ skill trong `.agents/skills`): `nature-writing/references/experiments.md` (ba câu hỏi cốt lõi, gói ablation, checklist chặt chẽ) và `venue-templates/references/ml_conference_style.md` (danh mục bằng chứng thực nghiệm cho hội nghị ML/vision gồm CVPR). Ghi chú của người dùng: **ablation làm sau vì còn ít thời gian**.

## Ba câu hỏi và nơi trả lời

| Câu hỏi của reviewer | Thí nghiệm của mình | Trạng thái |
| --- | --- | --- |
| 1. Có hơn baseline mạnh, gần đây, cùng giao thức không? | E1 (render hai view) và E3 (hình học) trên test dataset 8, 9: Endo-E2E-GS (train lại + hai checkpoint SCARED-C), MVSplat, DepthSplat (fine-tune cùng split), HK-Shi, teacher; cùng split, cùng metric, cùng thống kê | đang chạy (`ours_base`, `ours_refined` xong) |
| 1b. Có nhanh hơn không (nếu claim tốc độ)? | E6 theo Phụ lục 3: ours triển khai, ours eager, baseline như chạy trong đánh giá | chưa chạy, chạy sau cùng |
| 2. Module nào tạo ra gain (ablation)? | Bảng 2 hiện dùng số validation (zero-out từng phần, hàm tuyến tính, tinh chỉnh head, seed refiner, student GT/KD) | có số validation; ablation trên test để sau (ưu tiên thấp) |
| 3. Tổng quát hóa và giới hạn trong điều kiện khó hơn? | E4 dịch disparity (stress test hình học); E5 EndoNeRF zero-shot (miền khác, chỉ test); phân tích biên phải (mô tả thất bại); A4 nhiễu đầu vào trên validation | E4 đang chạy; E5 đang chạy; biên phải có số; A4 có số validation |

## Danh mục bằng chứng của `ml_conference_style.md` và cách mình đáp ứng

- **Dataset, nguồn gốc, giấy phép, split, loại trừ:** SCARED chia theo dataset (đối tượng) nên không rò rỉ giữa train, validation, test; loại dataset 4 và 5 (lệch dọc 17–22 px); EndoNeRF chỉ test, điều khoản dữ liệu được ghi. Cần ghi trong bài: HK-Shi không huấn luyện trên dataset 8, 9.
- **Baseline hiện đại và hợp lý, lý do chọn:** MVSplat, DepthSplat (feed-forward gần đây), Endo-E2E-GS (nội soi), HK-Shi và teacher (hình học). Chưa so với các phương pháp tối ưu theo từng cảnh vì khác bài toán; nêu rõ trong bài.
- **Ngân sách tinh chỉnh, tiêu chí dừng, chọn mô hình:** mọi baseline fine-tune theo một công thức cố định, checkpoint cuối, không chọn epoch. **Phải công khai** rằng refiner L6 được chọn trong thang 8 biến thể bằng một luật đăng ký trước trên 48 frame phát triển, và student là checkpoint cuối của lần chạy 10 epoch.
- **Metric và độ bất định, số lần chạy độc lập:** bootstrap theo frame và theo keyframe (5.000 lần). Phải nói rõ: student chính một seed; refiner có 3 seed (validation); bốn keyframe mỗi dataset.
- **Phần cứng, phần mềm, bộ nhớ, thời gian:** E6 cho độ trễ, bộ nhớ đỉnh đã ghi trong từng lần chạy; phần cứng ở Supplementary.
- **Ca thất bại và phạm vi chưa phủ:** Fig. 2 chọn frame theo luật cố định gồm ca tệ nhất; Limitations nói rõ biên phải, một benchmark, không có thí nghiệm khoảng cách dụng cụ tới mạch.

## Ưu tiên (còn ít thời gian)

- **P0, đang chạy hoặc đã lên lịch:** E1, E3, E4, E5, E6; Table 1 có tham số và độ trễ; dữ liệu cho Fig. 2 (cần một lượt suy luận nhỏ để lưu ảnh, chưa làm).
- **P1, rẻ và đáng làm:** chỉ số gần bài toán lâm sàng, ví dụ sai số khoảng cách giữa các cặp điểm 3D cách nhau 5–20 mm trên frame có ground truth (cần phụ lục giao thức viết trước); A4 nhiễu đầu vào đưa vào Supplementary từ số validation đã có.
- **P2, để sau:** ablation trên test (zero-out, hàm tuyến tính, head); seed thứ hai cho student; guard biên phải trên dữ liệu chưa thấy; thêm baseline.

## Kiểm tra theo checklist chặt chẽ của skill

1. Baseline gần đây và liên quan? Có (ba mô hình + hai hình học).
2. Metric đủ và chuẩn? Có (PSNR, SSIM, LPIPS hai view; EPE, MAE độ sâu; độ trễ).
3. Ablation gắn với mọi claim thiết kế? Phần lớn có trên validation; claim "kích thước là chính" chỉ trên validation (ghi rõ trong bài).
4. Claim ở Abstract và Intro có số hỗ trợ? Các claim về tốc độ và chất lượng so với baseline đang là `\TODO` có điều kiện, chờ E1 và E6.
5. Giới hạn phạm vi đánh giá được nêu? Có trong Conclusion.
