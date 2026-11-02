# Dự định phác thảo paper CVPR 2027 — 2026-11-01

**Trạng thái: đề xuất dàn ý nội bộ, chưa phải manuscript hay quyết định claim của tác giả.** Tài liệu này tổ chức lại lập luận trong file năm câu hỏi, đối chiếu template trong `paper/`, ledger claim và các kế hoạch thí nghiệm hiện có. Không chạy thêm thí nghiệm, không mở dữ liệu niêm phong/final_test/dev-48 và không sửa template hay các báo cáo cũ.

**Chỉ đạo cập nhật của tác giả: không lấy kết quả âm làm claim hoặc contribution.** Hướng bài tập trung vào phương pháp và lợi ích đã đo được của pipeline stereo student cùng Gaussian refinement. Chẩn đoán baseline, đối chứng không kết luận được và điều kiện chưa đạt thuộc phần báo cáo thí nghiệm/giới hạn; không dùng chúng để đặt title hoặc xây câu chuyện đóng góp. Số liệu gốc được giữ nguyên.

**Rà soát giọng văn và trạng thái (2026-11-01, chiều).** Bản này đã được chỉnh theo hai chỉ đạo của tác giả: (1) trong manuscript, nêu kết quả đo được rồi phân tích, không dùng nhãn tiêu chí nội bộ (N3, verdict, STOP, B1/C2, dev/confirmatory); các nhãn đó chỉ còn trong ghi chú nội bộ. Mọi con số giữ nguyên, hạn chế được nêu theo từng claim ở Discussion, không rải khắp bài. (2) Cập nhật trạng thái thí nghiệm: huấn luyện MVSplat/DepthSplat và phiên đánh giá test đang chạy tự động; MSDESIS không có weights; HK-Shi (Loo5) đã thêm vào trục hình học.

**Sửa theo hướng dẫn của giảng viên (2026-11-01, tối).** Mục 5 được viết lại theo cấu trúc của hướng dẫn: Method 3.1–3.4 (bài toán, tổng quan, từng khối, loss), Experiments 4.1–4.5, Discussion tách riêng (~nửa trang) và Conclusion. Ngân sách trang phân bổ lại: Method 2,85 và Experiments 2,45 (trước đó 2,0 và 3,5). Mục 4.5 là chi phí tính toán vì chưa có tác vụ phía sau nào được đo; không có Keywords vì template CVPR 2027 không có. Bản trước khi sửa: `/tmp/outline_before_guide_revision.md`.

Định tuyến writing: `task=manuscript` ở mức **outline**; `paper_type=research`; venue CVPR; tài liệu kế hoạch bằng tiếng Việt, manuscript dự kiến bằng tiếng Anh. Các số dưới đây được **trích từ tổng kết nội bộ**, chưa được tính lại từ từng frame trong công việc này. Việc đối chiếu artifact gốc và xác nhận nguồn của tác giả vẫn cần làm trước khi đưa vào bản nộp.

## 1. Hướng bài đề xuất

**Lập luận một câu:** Một pipeline feed-forward kết hợp stereo student học từ teacher với Gaussian attribute refinement nhỏ cho phép tái dựng hình học metric mà không dùng GT disparity miền đích làm target huấn luyện student, đồng thời cải thiện right-view PSNR trên ba student đã thử; ablation chỉ ra scale residual là thành phần chính tạo gain.

Trọng tâm dự kiến là **phương pháp tái dựng và các lợi ích đo được**, được kiểm tra bằng baseline và ablation. Hai đóng góp chính là recipe stereo student và refiner nhỏ; mức mới của tổ hợp/thiết kế cần được đối chiếu prior work, không tự nhận stereo-to-Gaussian hoặc distillation là ý tưởng đầu tiên. Tên ReliableEndo-GS là tên làm việc của repository; chưa dùng từ “reliable” như kết luận về độ an toàn lâm sàng, uncertainty calibration hay robustness tổng quát. Các giả thuyết ProbStereo-EndoGS/RiskRoute-GS trong chương trình nghiên cứu không tự trở thành kết quả của paper này; dàn ý không thay đổi gate hoặc phê duyệt một pivot.

Tên bài tạm đề xuất, để tác giả chọn:

1. **Stereo Distillation and Scale Refinement for Feed-Forward Endoscopic Gaussian Splatting** — tên đề xuất chính, nêu hai thành phần của phương pháp.
2. **Teacher-Supervised Stereo Reconstruction with Gaussian Scale Refinement** — nhấn supervision và refinement.
3. **Feed-Forward Endoscopic Reconstruction via Stereo Distillation and Gaussian Refinement** — nhấn đầu ra tái dựng từ một cặp stereo.

Không đưa “state of the art”, “first”, “validated”, “generalizable across surgery” hoặc “clinically reliable” vào title/abstract khi chưa có bằng chứng tương ứng. Khi so sánh với baseline, chỉ nêu khác biệt đo được và protocol; student GT-only cùng kiến trúc là đối chứng cho phần giám sát. [E1, E2]

### Năm câu hỏi dẫn đường

| Câu hỏi tác giả đã đặt | Câu trả lời paper dự định xây dựng | Nơi trả lời |
| --- | --- | --- |
| Làm gì? | Một cặp RGB stereo đã chỉnh hình và calibration → disparity/depth metric, trường Gaussian và ảnh render ở hai camera đầu vào; độc lập từng frame | Abstract; §1; §3.1 |
| Vì sao cần? | Cần tái dựng metric từ stereo với supervision có thể lấy từ teacher và một biểu diễn Gaussian cho chất lượng hai input views | §1; §3 |
| Làm như thế nào? | Student stereo học teacher đóng băng và loss ảnh hai view; lift hình học chuẩn; refiner residual nhỏ trên thuộc tính Gaussian | §3; Fig. 1 |
| Có hiệu quả không? | Đánh giá geometry, input-camera rendering, so sánh output chưa refine với output đã refine và chi phí; matched controls và shift response kiểm tra từng thành phần | §4.2–4.5; Tables 1–2, Fig. 2 |
| Đóng góp là gì? | Recipe stereo student không dùng GT disparity miền đích làm target và refiner nhỏ có gain chủ yếu từ scale; bằng chứng định lượng đi kèm phạm vi đã thử | Cuối §1; §5 |

## 2. Đã đọc template và skills nào

### Template trong `paper/`

Đã đọc `README.md`, `main.tex`, `preamble.tex`, `cvpr.sty`, các file `sec/0_abstract.tex`, `sec/1_intro.tex`, `sec/2_formatting.tex`, `sec/3_finalcopy.tex`, `sec/X_suppl.tex`, cùng cấu trúc bibliography/rebuttal. Đây vẫn là **template mẫu**: title, tác giả, nội dung hướng dẫn, bảng ví dụ và tài liệu tham khảo mẫu chưa phải paper của dự án. `main.tex` dùng chế độ `review`, khổ letter, 10 pt, hai cột và bibliography số `ieeenat_fullname`.

README chính thức của [CVPR author kit](https://github.com/cvpr-org/author-kit) cũng ghi cập nhật cho CVPR 2027; điều này không chứng nhận bản local giống từng byte với bản phát hành hiện hành. [README nguồn](https://raw.githubusercontent.com/cvpr-org/author-kit/main/README.md), kiểm tra 2026-11-01.

Theo [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines), kiểm tra 2026-11-01: nội dung chính tối đa **8 trang**, gồm hình/bảng; references và **LLM Usage Statement** được ngoài giới hạn. Statement phải đặt sau main text, trước references, nêu công cụ và phạm vi dùng. Bản review phải ẩn danh; không đổi font/margin hoặc dùng link để mở rộng nội dung. Đây là căn cứ cho ngân sách trang bên dưới.

Hai việc kỹ thuật cần kiểm tra khi thực sự dựng LaTeX: `main.tex` hiện có dòng `\\` trước phần khai báo; `main.bib` vẫn có citation giả của template. Chưa biên dịch nên **chưa kết luận PDF đạt chuẩn**. Không sửa những file này trong nhiệm vụ viết dàn ý.

### Skills đã đọc và cách áp dụng

| Skill đã cài | Cách dùng trong tài liệu này |
| --- | --- |
| [nature-writing](../.agents/skills/nature-writing/SKILL.md) | Một lập luận trung tâm; mỗi đoạn một việc; cấu trúc theo bằng chứng; phân bổ main text/SI; thuật ngữ nhất quán |
| [scientific-writing](../.agents/skills/scientific-writing/SKILL.md) | Gắn claim với nguồn; phân biệt quan sát, suy luận và dự định; giữ kết quả âm; tác giả kiểm tra số liệu/citation trước bản nộp |
| [venue-templates](../.agents/skills/venue-templates/SKILL.md) | Đọc kit local và kiểm tra nguồn CVPR 2027; lập ngân sách 8 trang; chưa nhận đã kiểm tra layout |
| [nature-polishing](../.agents/skills/nature-polishing/SKILL.md) | Đã đọc để phân biệt phạm vi; chỉ thích hợp khi đã có prose cần chỉnh. Chưa áp dụng quy trình polishing |
| [researchwrite](../.agents/skills/researchwrite/SKILL.md) | Đã đọc để phân biệt research proposal với manuscript. Không chạy quy trình proposal hay chấm điểm QA |
| Hướng dẫn viết của giảng viên (`paper/tmp/guide.docx`, thực chất là văn bản UTF-8) | Quyết định cấu trúc mục, khung abstract năm câu 150–200 từ, khung từng khối Method, thì và từ khẳng định; mâu thuẫn với template/skill được ghi ở §5 |

Đã đọc các core/fragments tương ứng của nature-writing, đặc biệt [Main-Text Discipline](../.agents/skills/nature-shared/core/main-text-discipline.md): kết quả làm thay đổi kết luận phải ở main text, kể cả kết quả bất lợi. Tham chiếu quy trình theo yêu cầu của scientific-writing/venue-templates: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065v2](https://arxiv.org/abs/2609.00065v2), [DOI](https://doi.org/10.48550/arXiv.2609.00065). Đã kiểm tra trang arXiv ngày 2026-11-01. Đây là nguồn cho **quy trình soạn tài liệu**, không phải baseline hay đóng góp khoa học của ReliableEndo-GS.

## 3. Những chỗ cần thống nhất trước khi viết prose

| Điểm | Quy ước/cách xử lý trong draft |
| --- | --- |
| **Ảnh phải có là input không?** | **Có.** `StereoStudent` dùng cả `batch.left` và `batch.right`; geometry prior nhận RGB trái. Câu “ảnh phải không bao giờ là đầu vào” trong file năm câu hỏi cần sửa ở bản manuscript: **cả hai ảnh là input cho stereo; màu Gaussian lấy từ ảnh trái**. Không sửa file nguồn trong nhiệm vụ này. [E1, E6] |
| Novel view hay input-camera reconstruction? | Hai camera chấm điểm chính cũng là hai camera đầu vào. Gọi **input-camera reconstruction**; có thể bàn cross-view consistency, nhưng không gọi đây là đánh giá held-out novel-view synthesis. |
| “Không cần GT” nghĩa gì? | Student KD không dùng GT disparity của miền đích làm target huấn luyện; teacher có lịch sử pretraining riêng cần công bố. Không suy ra toàn bộ hệ thống/dữ liệu/range selection đều GT-free. Baseline fine-tuning dự kiến dùng train GT để chốt depth range. [E5] |
| Đơn vị và hệ camera | `d = u_left - u_right`, pixel lưới A00 512×640; metric trong mô hình bằng mét, depth MAE báo bằng mm; Gaussian trong hệ camera trái. Nêu convention K/T và principal-point offset theo implementation chuẩn, không viết lại công thức xấp xỉ `fB/d` bỏ qua offset. |
| Tên refiner | L6 là MLP 11→32→32→7, **1.671 tham số**, dự đoán residual rotation/scale/opacity. Ablation cho thấy gain chủ yếu từ **scale residual trong các thử nghiệm đã làm**; không đổi tên kiến trúc thành scale-only hoặc nhận sửa được cả ba loại lỗi. [E1, E2] |
| Cohort 56 / 55 / 48 | Protocol dev là 56 frame `val_7`; một số tổng hợp hình học có **55 frame hợp lệ**. Phải truy vết nguyên nhân/denominator trước bảng cuối. Ablation lịch sử trên **48 frame** là cohort khác, không gộp vào 56; không mở lại 48 frame trong nhiệm vụ này. [E1–E3] |
| Đã test hay chưa? | Phần so sánh refiner với output chưa refine trên sequence 8/9 đã có kết quả (xem §4). Bảng so sánh base model trên final_test (E1), thử nghiệm dịch disparity (E4) và đo tốc độ (E6) **đang chạy tự động** (chờ huấn luyện MVSplat/DepthSplat xong); số sẽ được điền sau khi phiên đánh giá hoàn tất. Không sao chép checklist cũ "confirmatory chưa chạy" trong ledger. [E1, E2] |
| Cấu hình chính | B1 là đường chính trong tổng kết confirmatory mới, C2 thứ cấp; tài liệu FPS ngày trước còn trạng thái chưa chọn. Khi viết lịch sử lựa chọn phải đối chiếu frozen record, không tự sửa lời đăng ký cũ. [E1, E2, E4] |
| Dữ liệu đã mở | Các con số confirmatory dưới đây chỉ đọc từ tổng kết tác giả đã có. Không mở raw sealed-64/final_test. Dữ liệu final đã xem không còn là phép kiểm định mới cho guard phát triển sau đó. |
| Đơn vị khái quát hóa | SCARED 8 và 9 là **hai sequence trong cùng benchmark**, không phải hai bộ dữ liệu độc lập. Bốn keyframe là thông tin đã ghi cho val_7; số cụm của từng tập khác cần kiểm tra, không tự suy ra. |
| Baseline parity | Endo-E2E-GS retraining và released checkpoint phải tách dòng. **Native parity chưa được xác minh**; kết luận giới hạn ở checkpoint/protocol đã thử, không quy thành thất bại của paper gốc. |

Các bất nhất trên là ghi chú để tác giả đối chiếu, không phải thay đổi scientific formulation hay sửa artifact cũ. Phần viết được ngay không phụ thuộc vào việc giải quyết toàn bộ thí nghiệm còn thiếu.

## 4. Đóng góp dự kiến và bằng chứng hỗ trợ

**Status “có số trong tổng kết” dưới đây không có nghĩa đã độc lập tái tính hoặc được tác giả xác nhận từng artifact trong nhiệm vụ này.** Dùng nhãn dev/confirmatory/post hoc ở caption và văn bản liên quan.

### Hai contribution theo hướng phương pháp

Dùng nhãn **P1/P2** trong dàn ý cập nhật để không đổi nghĩa các ID **C1–C4** của ledger lịch sử. C1 chẩn đoán trong ledger không còn được đề xuất là contribution chính của manuscript.

| Contribution dự kiến | Bằng chứng hiện được ghi | Cách viết được đề xuất | Phạm vi |
| --- | --- | --- | --- |
| **P1 — Teacher-supervised stereo-to-Gaussian recipe** | Student KD dùng teacher disparity và loss ảnh hai view, không dùng target-domain GT disparity làm training target. Trên dev, student EPE **3,62 px**, teacher **3,19 px**. [E1, E2, E6] | “We present a teacher-supervised stereo-to-Gaussian pipeline that learns a target-domain stereo student without disparity ground-truth targets.” | Recipe trong nội soi; kết quả dev ở checkpoint đã ghi. Không nhận KD tốt hơn GT hoặc là phát minh mới của distillation. |
| **P2 — Small Gaussian refinement** | L6 **1.671 tham số**; dev right-PSNR gain **+0,99/+1,16/+0,92 dB** trên ba student. Ablation cho thấy gain chủ yếu từ scale residual. [E1, E2] | “A 1,671-parameter feed-forward attribute refiner improves right input-camera PSNR on the three tested students, with gains primarily attributable to scale residuals.” | Ba student đã thử, cùng benchmark; không gọi là universal refiner hay bảo đảm mọi vùng ảnh đều cải thiện. |

### Kết quả triển khai hỗ trợ

Báo cáo FPS cũ ghi B1 **55,5 FPS**, C2 **61,4 FPS**, 512×640; marginal refiner B1 **0,81 ms**. Đây là bằng chứng triển khai riêng hỗ trợ tính khả dụng của pipeline, không tự tách thành đóng góp thuật toán hoặc claim nhanh hơn baseline. Ghi đúng hardware/version/điều kiện đo; latency ở hash cuối chưa được đo lại do GPU shared, và không áp FPS này cho guard. [E4]

### Phân tích hỗ trợ trong Experiments (không nằm trong danh sách contribution)

| Kiểm tra | Số liệu được giữ nguyên | Vị trí và vai trò |
| --- | --- | --- |
| Geometry và shift response của baseline | Trên validation: baseline train lại EPE **15,47 px**, student **3,62**, teacher **3,19**; độ dốc shift của baseline **−0,01/0,01**, student **0,99/1,01** (train/val). Checkpoint SCARED-C giai đoạn 1 phát hành: EPE khoảng **11,3 px**. [E1, E2] | Bảng so sánh và hình diagnostic trong Experiments, mô tả bằng số (EPE, độ dốc), kèm ghi chú: baseline được train lại theo protocol của chúng tôi, native parity chưa xác minh. |
| Matched KD-vs-GT | 39.450 exposures, một seed: EPE **3,45 vs 4,17 px** (r = 1,21), depth MAE cùng chiều. [E1, E2] | So sánh matched một seed cho phần attribution; EPE thấp hơn của student KD được báo như một phép đo, không dùng làm luận cứ ưu thế so với GT. Không hoán đổi checkpoint KD 3,62 và matched KD 3,45 px. |
| Refiner trên các sequence giữ riêng (7 sealed, 8, 9) | Right-PSNR: sequence 8 **+0,599 dB**, sequence 9 **+1,328 dB**; tỉ lệ MSE ở 80 cột cuối view phải so với output chưa refine: **0,986** (sequence 8) và **1,054** (sequence 9). [E1, E2] | Bảng per-sequence ở main text: gain và tỉ lệ cột cuối được báo như hai đại lượng đo, theo sau là phân tích theo cột (post hoc). Không dùng nhãn tiêu chí nội bộ trong manuscript. |
| Scale ablation và linear control | Bỏ scale residual: **+0,99 → −0,19 dB**; bỏ rotation/opacity gần như không đổi. Linear control mạnh trên cohort lịch sử riêng. [E1–E3] | Kiểm tra thành phần tạo gain cho P2; đối chứng đủ để giới hạn độ phức tạp cần thiết. |
| Edge hypothesis/guard | Phân tích post hoc; tỉ lệ MSE cột cuối trên validation khoảng **1,028→0,995**, chưa có latency hay xác nhận trên dữ liệu mới. [E1, E2] | Exploratory analysis/limitations; không contribution, không nhập vào primary frozen method. |

Đóng góp kỹ thuật dự định là **P1–P2**. Shift test là công cụ kiểm tra hỗ trợ; provenance, preregistration và báo cáo đầy đủ là thực hành đánh giá. Không lấy baseline failure, failed criterion, selector NO-GO hoặc một đối chứng inconclusive làm contribution. Không đưa selector/router/uncertainty propagation vào abstract như thành phần đã được chứng minh.

## 5. Bố cục manuscript dự kiến — theo hướng dẫn của giảng viên và quy định CVPR 2027

Cấu trúc bám hướng dẫn viết của giảng viên (`paper/tmp/guide.docx`): Title, Abstract, Introduction, Related Work, Proposed method, Experimental results, Discussion, Conclusion, References. Theo template CVPR 2027, mục đánh số Ả Rập và **không có khối Keywords** (bỏ trừ khi giảng viên yêu cầu). Ngân sách 8 trang gồm cả hình/bảng; references và LLM Usage Statement nằm ngoài giới hạn. Đây là ước lượng bố cục, không phải kết quả biên dịch.

| Phần | Trang dự kiến | Nội dung chính |
| --- | ---: | --- |
| Abstract (150–200 từ) | 0,25 | năm câu, viết cuối |
| 1. Introduction | 1,00 | bối cảnh → phương pháp hiện có → khoảng trống → phương pháp đề xuất → đóng góp → lộ trình bài |
| 2. Related Work | 0,65 | 2.1 theo nhóm phương pháp; 2.2 Background |
| 3. Proposed Method | 2,85 | 3.1 bài toán 0,35; 3.2 tổng quan + Fig. 1 0,40; 3.3 ba khối 1,55; 3.4 loss 0,40; chi tiết huấn luyện 0,15 |
| 4. Experiments | 2,45 | 4.1 setup 0,40; 4.2 định lượng 0,75; 4.3 định tính 0,45; 4.4 ablation 0,55; 4.5 chi phí tính toán 0,30 |
| 5. Discussion | 0,50 | ý nghĩa, phạm vi, chi phí, tổng quát hóa, hạn chế dữ liệu |
| 6. Conclusion | 0,30 | đã làm gì; hạn chế và hướng tương lai cụ thể |
| **Tổng** | **8,00** | Cắt chi tiết triển khai sang SI trước khi nén font hoặc bỏ qualification |

### Quy tắc văn phong áp dụng toàn bài (từ hướng dẫn của giảng viên và skill)

- Thì hiện tại cho kiến thức chung và mô tả phương pháp; thì quá khứ cho kết quả thực nghiệm.
- Câu đơn giản, từ đơn giản; không dùng từ cảm xúc (amazing, good, very good); ít viết tắt, mỗi viết tắt định nghĩa một lần.
- `demonstrate` chỉ khi có bằng chứng trực tiếp; `suggest` khi kết quả có nhưng yếu; `may` cho khả năng chưa kiểm chứng.
- Ký hiệu toán học nhất quán qua mọi mục (một bảng ký hiệu duy nhất); mọi công thức, hình, bảng được tham chiếu trong văn bản; hình rõ nét ở khổ cột cuối.
- Mọi ý về phương pháp khác phải kèm trích dẫn; chỉ trích dẫn bài đã tra có thật và mở được; không dùng nhãn tiêu chí nội bộ (xem ghi chú đầu file).

### Title — viết cuối

Ngắn, phản ánh nội dung, không quá chung chung, không dùng "novel" hay "state-of-the-art". Ba phương án ở mục 1; chốt sau khi có kết quả.

### Abstract — viết cuối, 150–200 từ, năm câu

1. **Bối cảnh:** tái dựng metric từ một cặp stereo nội soi và vai trò của biểu diễn Gaussian.
2. **Vấn đề:** hạn chế của các phương pháp hiện có, nêu trung lập (cần giám sát disparity hoặc tối ưu theo từng cảnh).
3. **Phương pháp:** student stereo học từ teacher đóng băng, dựng Gaussian theo hệ mét, refiner nhỏ trên thuộc tính.
4. **Kết quả:** **một** con số định lượng tiêu biểu, có dataset, metric và đối chứng; chọn sau khi E1 xong (ứng viên: EPE của student so với đối chứng trên test, hoặc mức tăng right-PSNR của refiner).
5. **Ý nghĩa:** "these results demonstrate/suggest …" đúng mức bằng chứng.

Lỗi cần tránh (theo giảng viên): chỉ nêu đề tài mà không nêu phương pháp; thiếu ý nghĩa; viết "kết quả rất tốt" thay cho số; ý nghĩa không rút ra được từ kết quả; quá nhiều viết tắt.

### 1. Introduction — mỗi đoạn một việc

1. **Research background:** từ bối cảnh phẫu thuật nội soi đến bài toán một cặp stereo → hình học metric và ảnh render; không kể lịch sử bài toán. Động lực lâm sàng cần citation sơ cấp.
2. **Existing approaches:** phương pháp A làm gì, phương pháp B làm gì (không mô tả chi tiết); mỗi ý có citation (tối ưu theo từng cảnh; feed-forward Gaussian tổng quát; Endo-E2E-GS; stereo matching nội soi).
3. **Research gap:** các hạn chế của những phương pháp đó, viết trung lập (cần giám sát disparity của miền đích, hoặc tối ưu lại cho mỗi cảnh, hoặc chưa có bước hiệu chỉnh thuộc tính Gaussian cho render hai view).
4. **Proposed method (đoạn riêng):** ý tưởng tổng quan để giải khoảng trống; nói rõ cả hai ảnh là đầu vào stereo, màu Gaussian lấy từ ảnh trái, teacher chỉ dùng khi huấn luyện.
5. **Main contributions:** (i) recipe student stereo không dùng GT disparity miền đích làm target; (ii) refiner 1.671 tham số cải thiện render hai view, gain chủ yếu từ residual scale. Không nhận stereo-to-Gaussian hoặc distillation là ý tưởng đầu tiên.
6. **Roadmap:** một câu nói Mục 2 mô tả gì, Mục 3 trình bày gì, v.v.

### 2. Related Work

- **2.1 Nghiên cứu liên quan, chia theo nhóm phương pháp.** Với mỗi nhóm: các ý chính/khối chính của phương pháp → ý tưởng → ưu điểm → hạn chế (viết trung lập, ghi phạm vi của hạn chế). Ba nhóm: (a) tái dựng/render nội soi theo từng cảnh (EndoNeRF, EndoGaussian, EndoGS, G-SHARP, Free-DyGS và các bài đã tra); (b) Gaussian feed-forward (tổng quát: pixelSplat, MVSplat, DepthSplat, Splatt3R, GPS-Gaussian; nội soi: Endo-E2E-GS); (c) stereo matching nội soi và foundation stereo (HK-Shi, FoundationStereo), distillation.
- **2.2 Background:** chỉ nêu vì sao dùng từng công nghệ, không nêu nhược điểm, chi tiết trỏ sang citation: hình học stereo đã chỉnh hình và disparity → depth; 3D Gaussian splatting và render; chưng cất tri thức từ teacher stereo đóng băng.
- Citation khởi đầu lấy từ báo cáo khảo sát; tác giả đối chiếu từng bài trước `main.bib`. Không copy citation giả của template; EndoVGGT đã withdrawn không được dùng.

### 3. Proposed Method (cấu trúc theo hướng dẫn)

- **3.1 Problem formulation:** đầu vào (cặp RGB `[0,1]` 512×640 đã chỉnh hình, `K_L`, `K_R`, `T_R_from_L`, support mask), đầu ra (disparity `d = u_left − u_right` theo pixel, depth metric theo mét, trường Gaussian trong hệ camera trái, ảnh render ở hai camera đầu vào), mục tiêu tối ưu, giả thiết (một frame độc lập, không tối ưu theo cảnh, camera chỉnh hình). Có công thức. Gọi đây là input-camera reconstruction, không phải novel-view synthesis.
- **3.2 Overview:** Fig. 1; đầu vào/đầu ra, các khối chính, tô đậm các thành phần đề xuất; phân biệt khối đóng băng và khối huấn luyện.
- **3.3 Các khối (mỗi khối trả lời: đầu vào, đầu ra, giải quyết vấn đề gì, hoạt động ra sao, vì sao hiệu quả, khác gì cái hiện có):**
  - **3.3.1 Teacher-supervised stereo student:** student đa tỉ lệ dùng cả hai ảnh; teacher FoundationStereo đóng băng chỉ dùng khi huấn luyện.
  - **3.3.2 Metric Gaussian construction:** disparity → depth → tâm 3D có tính đến điểm chính; lưới lấy mẫu cố định trên view trái; màu từ ảnh trái; thuộc tính khởi tạo theo quy tắc kích thước.
  - **3.3.3 Small attribute refiner:** MLP 11→32→32→7, 1.671 tham số, residual rotation/scale/opacity có biên, ghép bằng hàm chuẩn; residual bằng 0 trả về output chưa refine. Lý do nhấn scale đến từ ablation, không đổi mô tả sau khi xem kết quả.
- **3.4 Loss functions:** loss của student (SmoothL1 disparity theo teacher + Charbonnier tái dựng hai view, trọng số theo config) và loss của refiner (Charbonnier trung bình hai view); nêu vì sao chọn từng loss, vai trò và trọng số; ký hiệu nhất quán.
- Chi tiết triển khai ngắn (B1/C2) ở cuối mục; CUDA graph và kernel biên dịch sang SI. Biến thể giữ ảnh chưa refine ở dải cạnh phải không thuộc phương pháp chính (xem Discussion/SI).

### 4. Experiments

- **4.1 Experimental setup:** dữ liệu (SCARED A00, train 1/2/3/6, validation 7, test 8/9; lý do loại 4/5), metric (PSNR/SSIM/LPIPS hai view, EPE, depth MAE/RMSE; số frame hợp lệ và cách gộp), môi trường (phần cứng, framework, phiên bản), cấu hình huấn luyện (optimizer, learning rate, batch size, số bước), baseline và cách mỗi baseline được chạy (train lại, phát hành, fine-tune theo protocol).
- **4.2 Kết quả định lượng (Table 1):** hàng = student chưa refine/đã refine, các student GT-only và KD matched, Endo-E2E-GS (train lại; checkpoint SCARED-C do dự án huấn luyện), MVSplat/DepthSplat fine-tune, HK-Shi, teacher; cột = EPE, depth MAE, PSNR/SSIM/LPIPS hai view, số tham số. Nêu chỉ số nào cải thiện, mức cải thiện, kết quả có nhất quán giữa sequence 8 và 9 không. Báo cả tỉ lệ MSE 80 cột cuối view phải theo từng sequence như một đại lượng đo. Độ dốc shift disparity là một panel nhỏ.
- **4.3 Kết quả định tính (Fig. 2):** vài ví dụ chọn theo luật cố định (trung vị, tệ nhất, dải cạnh), render và disparity so với ground truth (ảnh đầu vào và GT disparity).
- **4.4 Ablation (Table 2):** bỏ từng thành phần của refiner (rotation/scale/opacity), refiner so với đối chứng tuyến tính và đầu thuộc tính tinh chỉnh, matched KD-vs-GT; mỗi hàng ghi rõ cohort và số seed.
- **4.5 Chi phí tính toán:** số tham số, bộ nhớ đỉnh, thời gian chỉ khi đo trên GPU rảnh, kèm điều kiện; ghi "not measured" nếu chưa đo được. Hiện không có tác vụ phía sau (downstream) nào được đo nên không có mục downstream.

### 5. Discussion (khoảng nửa trang)

Trả lời theo hướng dẫn: kết quả tốt hay không, vì sao; có đúng với mọi trường hợp không (gain nhỏ hơn ở dải cạnh phải của sequence 9, tỉ lệ 1,054; giải thích theo cột là phân tích post hoc, cơ chế về support của view trái là giả thuyết); chi phí tính toán và bộ nhớ; khả năng tổng quát hóa (hai sequence thuộc cùng benchmark SCARED, chưa có dữ liệu ngoài); hạn chế của dữ liệu (số sequence độc lập ít, calibration, sequence 4/5 bị loại).

### 6. Conclusion

Làm được gì: phương pháp, bài toán giải quyết, kết quả chính trong phạm vi đã đo. Limitations và future work **cụ thể**: dữ liệu ngoài có calibration độc lập, thêm seed và thêm sequence cho phần KD, refiner trên backbone khác, biến thể nhận biết dải cạnh; không viết chung chung.

## 6. Hình/bảng dự định và phân bổ main text/SI

| Display | Nội dung và thông điệp | Nguồn/trạng thái |
| --- | --- | --- |
| **Fig. 1 — Overview** | Hai RGB vào stereo; teacher là nhánh training; disparity + calibration lift; màu Gaussian từ trái; L6 → render hai input cameras. Phân biệt frozen/trainable và input/supervision | Sơ đồ dự kiến từ E1/E6; chưa vẽ |
| **Fig. 2 — Qualitative comparison (§4.3)** | Render và disparity so với ground truth theo luật chọn cố định (trung vị, tệ nhất, dải cạnh) | Có số tổng kết; cần sinh plot từ artifact hợp lệ theo phạm vi được phép |
| **Table 1 — Comparisons** | Panel geometry và panel rendering; input, colour sources, pretraining/adaptation, checkpoint, metric eligibility và cohort. Released/retrained/fine-tuned tách dòng | Bảng test chính còn thiếu; không điền số giả |
| **Table 2 — Attribution/ablation** | Matched KD-vs-GT (r = 1,21, một seed); chưa refine/L6/bỏ scale. Nếu có ladder 48 frame thì panel riêng ghi rõ cohort | Số tổng kết sẵn; cần đối chiếu artifact trước bảng cuối |
| **Table 3 — Per-sequence rendering (§4.2; SI nếu thiếu trang)** | Per-sequence ΔPSNR/SSIM/LPIPS của refiner và tỉ lệ MSE ở 80 cột cuối view phải (sequence 9: 1,054), kèm khoảng tin cậy theo estimator thực tế; con số 1,054 vẫn nằm trong văn bản §4.2 và §5 dù bảng chuyển SI | Có tổng kết; không tái tính trong nhiệm vụ này |

Tối đa khoảng ba hình và ba bảng chính là mục tiêu bố cục, không phải yêu cầu của venue. Nếu tràn trang, rút panel không đỡ claim, giữ panel tỉ lệ cột cuối và matched KD-vs-GT. Dòng cost chỉ thêm khi có phép đo đủ điều kiện, có thể đặt gọn cạnh Table 1 thay vì thêm một bảng lớn.

| Kết quả/chi tiết | Phân loại | Vị trí dự định |
| --- | --- | --- |
| Geometry của student và input-camera rendering | `necessary_support` cho P1/P2 | §4.2; Table 1 |
| Shift response của checkpoint thử | `diagnostic_support`, không contribution | §4.2 (panel nhỏ trong Table 1) |
| Native parity chưa xác minh; GT student control | `necessary_support` / `qualification` | §4.1–4.3; caption Table 1 |
| KD matched dưới ngưỡng | `qualification` làm thay đổi claim | Main text + Table 2 |
| Scale ablation và linear control mạnh | `core_discovery` / `necessary_support` | §4.4; Table 2 |
| Tỉ lệ MSE ở 80 cột cuối view phải (sequence 9: 1,054) | `qualification` / `edge_case`, báo cùng gain | §4.2 và §5; Table 3; abstract nếu abstract trích gain trên sequence 8/9 |
| Seed/perturbation chi tiết, negative selector gate | `robustness` / negative evidence thứ cấp | SI với trỏ dẫn ngắn; không sửa Gate NO-GO thành success |
| Guard dải mép sau test | `exploratory` | Discussion ngắn; SI đầy đủ, giữ nhãn post hoc |
| Config, hash, checkpoint lineage, lệnh, camera conversions | `provenance_detail` | Methods/SI/reproducibility appendix; không để internal machine paths trong bản anonymous |

Main text giới thiệu claim ở §1, chứng minh ở §4, giải thích ở §5; không lặp lại toàn bộ số/CI ở cả prose lẫn caption. Vì đây là **outline mới**, chưa có đoạn Results cũ bị xóa hoặc word-count delta; bảng trên là allocation plan, chưa phải compression audit đã thực hiện.

## 7. Quy tắc xây bảng baseline

- **Geometry:** chỉ chấm disparity/depth đúng sign, grid và scale; teacher depth không tự có render. Mô hình monocular cần scale alignment phải có nhãn và protocol riêng; không GT scale fitting. Released endoscopy weights cần kiểm tra training overlap với **cả val_7 và test 8/9** trước khi gọi là unseen.
- **Rendering:** PSNR/SSIM/LPIPS trái/phải ở hai input cameras. Nêu colours từ một hay hai ảnh. Không gọi right-view score của MVSplat/DepthSplat là novel-view score khi ảnh phải cũng là input/colour source.
- **Generalization mode:** tách official checkpoint zero-shot, adaptation trên train_1236 và matched retraining. Fine-tune dự kiến 13.150 pairs ×3 = 39.450 exposures là khớp split/samples, **không** khớp architecture, optimizer, parameter count, pretraining compute hoặc colour sources. [E5]
- **Per-scene references:** chỉ Related Work hoặc bảng protocol khác có tiêu đề rõ; không trộn số EndoNeRF/EndoGaussian/EndoGS/G-SHARP với feed-forward unseen-frame scores.
- **Failure visibility:** mô hình bị chặn hoặc không chạy được phải có ghi chú khả thi, không âm thầm bỏ vì chất lượng. MVSplat và DepthSplat-small đang được fine-tune đủ 39.450 bước ở 512×640 (giải quyết bộ nhớ bằng gradient checkpointing, đã vá điểm chính của rasterizer gốc); HK-Shi (Loo5, hình học) đã thêm; MSDESIS không có weights chính thức truy cập được (HTTP 403) nên không đưa vào. [E5]

Mọi bảng working draft để ô `not run`, `not measured`, `not applicable` hoặc `pending verification` đúng lý do. Chỉ thay bằng số sinh từ metrics artifacts có provenance; không suy ra baseline score từ paper khác.

## 8. Có thể viết gì ngay; điều gì đang thiếu

### Viết v0 được ngay

Title/Introduction hướng P1/P2; task/conventions; mô tả train/KD/refiner theo config; Related Work có citation đã kiểm; Results dựa trên tổng kết với nhãn provisional. Shift test là diagnostic trong Experiments; các kết quả và limitations giữ đúng vai trò báo cáo, không contribution. Khung abstract ở trên dùng để kiểm tra câu chuyện, **abstract hoàn chỉnh viết sau Results**.

Thứ tự làm việc đề xuất: **evidence tables → Results → Methods → Introduction/Conclusion → Related Work/Discussion → title/abstract → kiểm tra layout/citation**. Đây là thứ tự soạn thảo, không thay thứ tự section của manuscript.

### Missing evidence / quyết định của tác giả cho các công việc tiếp theo

| Việc | Vì sao ảnh hưởng paper | Giới hạn hiện tại |
| --- | --- | --- |
| Bảng main test cho baseline/base models và shift diagnostic ngoài val_7 | Cơ sở cho phát biểu so sánh trên test | Đã được tác giả cho phép; E1/E3/E4 đang chạy tự động theo giao thức đã băm, kết quả điền vào Table 1 sau khi xong |
| MVSplat/DepthSplat adaptation | Tăng sức đối chiếu rendering đúng protocol | Đang huấn luyện (MVSplat khoảng 09:40, DepthSplat khoảng 17:20 ngày 1/11); render hai view của các mô hình này dùng màu từ cả hai ảnh, nên trục hình học là trục phân biệt chính |
| Matched photo-only arm | Tách KD khỏi kiến trúc và loss ảnh | Planned, chưa chạy; không thay bằng config lịch sử không matched |
| Cohort/estimator audit | 55 vs 56 frames, 48-frame ladder, frame dependence, CI theo sequence/keyframe | Đối chiếu metadata/artifacts chỉ trong phạm vi được phép; không tự nhận hàng nghìn frame là hàng nghìn ca độc lập |
| External calibrated evaluation | Hạn chế claim trong SCARED và kiểm định guard trên dữ liệu mới | EndoNeRF chưa chấp nhận terms, không download; baseline/calibration chưa tài liệu hóa, không suy scale từ GT. Depth STTR-Light chỉ là pseudo-reference. SERV-CT cũng cần kiểm tra availability/protocol trước |
| Fair runtime/version provenance | Chưa chứng minh nhanh hơn baseline; guard có thêm rasterization | GPU rảnh, cùng đường triển khai/phần cứng, final source hash; không lấy thời gian GPU shared |
| Human source/claim check | Nguồn tổng kết có mục cũ và số làm tròn | Tác giả xác nhận từng con số, frozen criteria/choices, citations và nội dung cuối; tác vụ outline không chứng nhận bước này đã xong |

Việc tải EndoNeRF (cần chấp nhận điều khoản), dữ liệu ngoài và hoạt động mới trên sealed data vẫn cần tác giả cho phép riêng. Dàn ý vẫn hoàn tất được với các giới hạn này. Nếu deadline không đủ, thu hẹp claim và ghi model không chạy được; không gọi thí nghiệm dự định là kết quả.

## 9. Chuyển sang LaTeX khi viết manuscript thật

Đây là **file map dự kiến**, chưa tạo hoặc sửa các file tương ứng:

| File | Công việc dự kiến |
| --- | --- |
| `paper/main.tex` | Thay title/inputs mẫu; giữ review mode và template formatting; kiểm tra dòng mở đầu; bố trí LLM Usage Statement trước bibliography |
| `paper/sec/0_abstract.tex` | Abstract tiếng Anh sau khi chốt kết quả |
| `paper/sec/1_intro.tex` | Introduction theo sáu đoạn ở trên |
| `paper/sec/2_related.tex` | 2.1 theo nhóm phương pháp, 2.2 Background |
| `paper/sec/3_method.tex` | 3.1 bài toán, 3.2 tổng quan, 3.3 ba khối, 3.4 loss |
| `paper/sec/4_experiments.tex` | 4.1 setup, 4.2 định lượng, 4.3 định tính, 4.4 ablation, 4.5 chi phí |
| `paper/sec/5_discussion.tex`, `paper/sec/6_conclusion.tex` | Discussion (khoảng nửa trang) và Conclusion tách riêng |
| `paper/sec/llm_usage.tex` | Statement đúng phạm vi AI đã dùng; tác giả bổ sung đầy đủ lịch sử dùng cho code/research/writing, không chỉ lần outline này |
| `paper/main.bib` | Thay references giả bằng citation gốc đã kiểm tra; không tự gán DOI/arXiv từ trí nhớ |
| Supplement riêng | Protocol/config/provenance, full ablations và exploratory notes; tách khỏi main PDF |

Không sửa `cvpr.sty`/bibliography style để ép trang. Build products và hình/bảng sinh tự động về sau đặt dưới artifact root D: theo quy tắc dự án; không tạo cache/env ở C:. Chưa cần build bản template mẫu để gọi manuscript đã sẵn sàng.

## 10. Nguồn và kiểm tra truy vết

| ID | Nguồn đã đọc | Dùng cho |
| --- | --- | --- |
| E1 | [paper_core_questions_2026-10-31.md](paper_core_questions_2026-10-31.md) | Năm câu hỏi, tổng kết số liệu, giới hạn và claim tác giả dự kiến |
| E2 | [paper_contributions_draft_2026-10-31.md](paper_contributions_draft_2026-10-31.md) | Claims ledger; cập nhật confirmatory, tỉ lệ cột cuối 1,054 và các mục còn cũ |
| E3 | [experiments_needed_2026-10-31.md](experiments_needed_2026-10-31.md) | Gaps, historical 48-frame ablation, việc cần tác giả quyết định |
| E4 | [fps_bc_status_2026-10-30.txt](fps_bc_status_2026-10-30.txt) | FPS riêng, equivalence, cấu hình và điều kiện/version chưa đo lại |
| E5 | [baseline_finetune_plan_2026-10-31.md](baseline_finetune_plan_2026-10-31.md) | Protocol adaptation, train-only range, capped OOM, order/projection blockers |
| E6 | [student.py](../src/reliable_endo_gs/reconstruction/student.py), [model.py](../src/reliable_endo_gs/reconstruction/model.py) | Đối chiếu cả hai ảnh là stereo input; geometry prior `rgb=batch.left` |
| E7 | [feasibility](baseline_feasibility_2026-10-31.md), [addendum](baseline_feasibility_addendum_2026-10-31.md), [run report](baseline_run_report_2026-10-31.md) | Các kết luận khả thi hiện có; không khảo sát/rerun mới trong nhiệm vụ này |
| V1 | [paper/main.tex](../paper/main.tex), [paper/README.md](../paper/README.md), [cvpr.sty](../paper/cvpr.sty), [formatting sample](../paper/sec/2_formatting.tex) | Cấu trúc local kit, chưa phải manuscript |
| V2 | [Author Guidelines 2027](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines), [author kit](https://github.com/cvpr-org/author-kit) | Kiểm tra nguồn venue ngày 2026-11-01; không chứng nhận local copy bằng upstream |

Snapshot SHA-256 của nguồn chính lúc lập dàn ý, để phát hiện thay đổi về sau; hash **không thay thế** xác minh nội dung:

```text
bb688bfc0062f2b36bcc274a306f905ce45689fd91950e54ff78338ffdfce2c7  docs/paper_core_questions_2026-10-31.md
d43f01011336b2b21ddb9281bdf5bfc332493362d0282f52369d08c589ca7f3f  docs/paper_contributions_draft_2026-10-31.md
7def2df6a89969db98a7635afc79b6023d59c55b66b155cfc46b423d9835c4b7  docs/experiments_needed_2026-10-31.md
0b7b627139fccf9d275b723ef2aaa8d0c43fce1333b9d4d8d17aef9bab8b1d83  docs/fps_bc_status_2026-10-30.txt
b28c87b4c0521dba74b96e67f58c0bb2839826f95a0100b1c3f488d3160b889e  docs/baseline_finetune_plan_2026-10-31.md
```

Không mở các `rows.jsonl`, worklist niêm phong hoặc ảnh/depth test để lập dàn ý này. Việc có aggregate số trong E1/E2 không cấp quyền đọc lại dữ liệu gốc.

## 11. Trạng thái bàn giao

- **Implemented:** dàn ý đã cập nhật theo chỉ đạo không claim negative, giọng văn trung lập và hướng dẫn của giảng viên: mục 5 là cấu trúc mới (Method 3.1–3.4, Experiments 4.1–4.5, Discussion và Conclusion tách riêng); title/abstract/contributions tập trung vào P1 stereo recipe và P2 refiner; các kết quả gốc vẫn được báo cáo trong Experiments và Discussion.
- **Tested:** kiểm tra tài liệu local, đối chiếu stereo-input/colour path trong source và kiểm tra trang venue công khai; kiểm tra link local, hash nguồn và tổng ngân sách trang khi hoàn tất file. Đây là kiểm tra tài liệu, không phải tái kiểm định kết quả khoa học.
- **Not tested:** chưa biên dịch LaTeX/PDF, chưa audit toàn bộ citation/artifact gốc, chưa tái tính metric hoặc CI, chưa dựng hình, chưa chạy test phần mềm vì chỉ thêm Markdown.
- **Blocked:** không có blocker đối với việc giao outline. Các claim test/baseline superiority, guard validation, fair speed và KD attribution mạnh hơn còn thiếu bằng chứng/authorization như bảng mục 8.
- **Planned:** tác giả chốt cách đặt trọng tâm, đối chiếu claim/citation với artifact được phép, rồi viết manuscript theo file map. Không có training, data download hay test access nào được lên lịch tự động từ tài liệu này.

**Vì sao chọn cấu trúc này:** Introduction và Method nêu giải pháp, Experiments cung cấp bằng chứng cho P1/P2, ablation xác định phần tạo gain và runtime mô tả chi phí thực tế. Diagnostics và giới hạn giúp đọc đúng các kết quả; không dùng chúng làm contribution. Các kết quả làm thay đổi phạm vi kết luận vẫn được báo cáo trong main text.

Để chỉnh hướng ở lượt tiếp theo, chỉ cần chỉ ra claim hoặc section cần đổi; những phần còn phù hợp có thể giữ nguyên.
