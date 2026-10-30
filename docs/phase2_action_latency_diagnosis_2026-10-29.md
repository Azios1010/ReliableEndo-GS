# Bao cao chan doan Phase II: action, latency, baseline - 2026-10-29

Nguoi doc: chu du an. Tac gia: Claude Code (phien lam viec ngay 2026-10-29). Trang thai: tham do (EXPLORATORY), chua pre-register, KHONG phai ket qua cua gate. 64 frame val_7 da niem phong va final_test (dataset 8, 9) chua bi dong toi.

Nhan bang chung dung trong bao cao: [DO] = do truc tiep tu file ket qua; [SUY LUAN] = suy ra tu so lieu da do; [GIA THUYET] = chua kiem chung.

## 1. Tom tat

- [DO] FPS 11 cua paper khop voi baseline o dung do phan giai goc: cung checkpoint, cung GPU, baseline chay 30 FPS o 512x640 nhung chi ~11 FPS o 1024x1280. Con so 29.5 FPS cua repo la so o 512x640, khong so duoc voi paper.
- [DO] KD cham hon baseline ngay ca khi khop do phan giai (xem muc 3). Repair khong cuu duoc chan toc do.
- [DO] Nguyen nhan KD cham (muc 3.1): GroupNorm 1 nhom o batch 1 ton ~47% thoi gian (35 ms) tren 1 khoi GPU, con TAT CA conv chi ~5 ms. Thay bang phep tinh song song tuong duong (khong train lai, EPE/PSNR khong doi): 512x640 KD tu cham 2.3 lan xuong 1.41 lan; 1024x1280 tu cham 3.0 lan xuong 1.21 lan (9.3 vs 11.2 FPS) - van cham hon baseline (muc 3.2). Sau sua GPU ranh ~48% (nghen o host: ~3100 kernel va 145 lan dong bo moi forward), con dia giam them nhung chua thu.
- [DO] Action patch 32x32 khong the co tac dung: moi can thiep chi ghi 0.31% khung hinh; sua hoan hao cung chi giam toi da ~0.29% loss toan khung.
- [DO] Khi phu rong, Gaussian action co tac dung that nhung la danh doi view: view trai tang manh, view phai giam. Chi ap dung o NUA TRAI cua anh thi ca hai view deu tot hon o 40/40 frame (val_7 va train).
- [DO] Mot hieu chinh HANG SO (khong mang) tai hien ~2/3 muc tang view trai va ~90% muc mat view phai: day la hieu chinh do day Gaussian theo phap tuyen, khong phai 'sua loi co hoc'.
- [DO] Baseline khong hong: fit train 0.59 px nhung val_7 14.2 px (khong tong quat qua sequence).
- [DO] Chat luong (muc 5.1): KD hon ro ret o hinh hoc va view giu lai; baseline cao hon o view trai NHUNG diem do khong doi khi lam sai do sau (26.89 dB du EPE tang 14 -> 19 px), nen khong do duoc chat luong 3D. Khong the noi 'KD hon moi mat'.
- [SUY LUAN] Action khong du lam contribution chinh (tran ~+1 dB hai view, them ~18 ms). Contribution ma du lieu ung ho nam o tong quat hoa qua sequence (EPE 3.6 vs 15.5 px) va chat luong view giu lai (right PSNR 22.1 vs 17.3 dB).

### Tra loi ngan cho 3 cau hoi cua ban

- Sua action roi chay lai hay doi method? Action patch 32x32: khong sua duoc, bo. Action phu rong chi o NUA TRAI anh: cai thien ca hai view o 40/40 frame, nhung tac dung nho va la hieu chinh do day Gaussian. Nen dua vao head student (can ban duyet), khong giu lam module luc suy luan; bo stereo repair va routing.
- Ket qua tot nhat co the: bang chung hien co cho tran ~+1 dB TB hai view (quy tac co dinh nua trai: view trai +0.85, view phai +0.44 dB; chon tot nhat tung frame them ~0.27 dB, so lac quan). Khong co bang chung cho muc cao hon.
- Co du lam contribution khong? Khong du lam contribution chinh. Luan diem toc do cung dang bi bac (muc 3). Huong du lieu ung ho: tong quat hoa qua sequence va chat luong view giu lai (muc 5, 6).

## 2. Trang thai cac run

```
Hang muc                             Trang thai                                                    
-----------------------------------  --------------------------------------------------------------
KD student                           Xong: final.pt, 131,500 buoc                                  
Baseline Endo-E2E-GS                 Xong: Stage1 100,100 + Stage2 10,000; benchmark 56 frame val_7
Stage A (repair, student dong bang)  Xong 13,150/13,150 mau; student/head/Adam khong doi (bitwise) 
Gate A                               NO-GO (03:28 29/10); collection, Gate B, selector KHONG chay  
final_test (dataset 8,9)             Chua mo                                                       
```

Nguon: docs/baseline_stage2_gpu0_restart_2026-10-28.md, docs/kd_latency_frozen_actions_report_2026-10-28.md, run phase2-frozen-actions-20261028T114155Z.

## 3. Latency va FPS

[DO] Giao thuc: batch 1, input nap san len GPU, tinh model + geometry + dung Gaussian + render CA HAI view; dong bo quanh moi lan goi; 10 warm-up, 100 lan do xoay vong 8 frame val_7 co dinh; GPU NVIDIA GeForce RTX 5070 Ti, torch 2.7.1+cu128. Khong tinh I/O, giai ma, H2D, metric. May khong danh rieng (job GPU1 cua ban van chay).

```
model     do phan giai  median ms  p90 ms  FPS(median)  FPS(mean)  VRAM MB  median lan 2
--------  ------------  ---------  ------  -----------  ---------  -------  ------------
baseline       512x640       32.9    34.8         30.4       30.2      460          32.7
kd             512x640       76.6    78.8         13.1       13.0      647          76.8
baseline     1024x1280       91.8    93.2         10.9       10.9     1824             -
kd           1024x1280      277.9   280.0          3.6        3.6     2543             -
```

Cot 'median lan 2' chi co o 512x640: do lai cuoi phien de kiem tra troi/nhieu. VRAM la dinh PyTorch, gom input da nap.

```
model     do phan giai  left PSNR  right PSNR  EPE px
--------  ------------  ---------  ----------  ------
baseline       512x640      26.89       17.94   14.21
kd             512x640      25.73       23.13    3.33
baseline     1024x1280      27.33       16.99   56.19
kd           1024x1280       7.11        7.11    7.53
```

Chat luong tren cung 8 frame do latency (de doc cung latency; KD chi duoc train o 512x640). Day khong phai benchmark chinh thuc.

[DO] Khi khop do phan giai: o 512x640 baseline 32.9 ms (30.4 FPS) va KD 76.6 ms (13.1 FPS), tuc KD cham 2.3 lan; o 1024x1280 baseline 91.8 ms (10.9 FPS, khop 11 FPS cua paper) va KD 277.9 ms (3.6 FPS), tuc KD cham 3.0 lan. Khoang cach toc do khong thu hep khi tang do phan giai ma rong ra.

[DO] Do lai cuoi phien o 512x640 cho 32.7 ms (baseline) va 76.8 ms (KD): on dinh, lech duoi 1% so voi lan dau.

[LUU Y] Ca hai mo hinh chi duoc train o 512x640. O 1024x1280 so do chat luong khong co y nghia: baseline EPE ~56 px, KD render ~7 dB PSNR (hong). Con so 1024x1280 chi noi ve chi phi tinh toan cua kien truc o kich thuoc do, khong phai kha nang cua mo hinh tai do phan giai do; paper dung mot mo hinh duoc train o do phan giai do.

[SUY LUAN] Luan diem toc do cao hien bi bac o ca hai do phan giai: KD cham hon baseline 2.3 lan (512x640) va 3.0 lan (1024x1280). Muon co loi the toc do phai thiet ke lai student (decoder GroupNorm toan do phan giai), khong the dat duoc bang repair.

[DO] So lieu benchmark chinh thuc truoc do (56 frame, 200 lan, 512x640): baseline median 31.95 ms (29.5 FPS), KD median 75.25 ms (11.8 FPS). Kiem tra rieng bang script phu (4 frame x 30 lan) cho baseline: 32.9 ms o 512x640 va 91.8 ms o 1024x1280 (10.9 FPS); tat/bat AMP khong doi dang ke.

### 3.1 Vi sao student cham du chi chay student

[DO] Ket luan ngan: student cham vi MOT toan tu - GroupNorm 1 nhom o batch 1 - khong phai vi model lon (1.45 trieu tham so). GroupNorm ton 34.7 ms trong 73.6 ms (47%), trong khi TAT CA cac lop conv cong lai chi 4.9 ms. Do trung vi 73.6 ms o day (lan do rieng cua script profile) lech nhe so voi 76.6 ms o bang tren (co the do khac thoi diem va may dung chung; chua xac dinh).

```
thanh phan (512x640, batch 1)                                      ms  % cua tong
---------------------------------------------------------------  ----  ----------
mang stereo cua student (tong)                                   50.9         69%
  decoder do phan giai day du (7->32 kenh + 3 khoi residual)     32.4         44%
  trich dac trung (chay 2 lan: trai va phai)                     12.6         17%
  bo dieu chinh cost 3D                                           1.7          2%
phan con lai: geometry prior, head, dung Gaussian, 2 lan render  22.8         31%
--- theo toan tu ---                                                             
GroupNorm (40 lan goi / lan forward)                             34.7         47%
tat ca conv (cudnn, 45 lan goi)                                   4.9          7%
```

Cac dong module/toan tu chong lan nhau nen khong cong thang. Mang stereo cua baseline (encoder + RAFT) chi ~9 ms (docs/kd_latency_frozen_actions_report_2026-10-28.md); phan con lai cua hai he thong cung cap ~23-28 ms. Tuc la hau het khoang cach 2.3 lan (~44 ms) nam o mang stereo cua student (~51 ms so voi ~9 ms).

```
GroupNorm tren tensor                                        ms / lan goi
-----------------------------------------------------------  ------------
[1,32,512,640], 1 nhom (batch 1: truong hop that)                    6.00
[4,32,512,640], 1 nhom (nhieu gap 4 lan cong viec)                   6.81
[8,32,512,640], 1 nhom (nhieu gap 8 lan cong viec)                   7.26
[1,32,512,640], 32 nhom (phep toan KHAC, chi de tham chieu)          0.30
[1,64,128,160], 1 nhom (lop dac trung)                               0.40
[8,64,128,160], 1 nhom                                               0.89
```

[DO] Nhan 8 lan khoi luong ma thoi gian chi tang 21%: thoi gian gan nhu khong phu thuoc batch. [SUY LUAN] Phu hop voi co che 'mot khoi cho moi (mau, nhom)': o batch 1 kernel thong ke (RowwiseMoments) chi dung ~1 trong 70 SM cua GPU; toi chua doc ma nguon kernel de xac nhan. Neu dung thi day la nghen do song song hoa cua toan tu, khong phai do so phep tinh.

```
thu nghiem (khong doi trong so, khong train lai)     median ms  tang toc  lech disparity max (px)
---------------------------------------------------  ---------  --------  -----------------------
bat cudnn.benchmark                                       75.2  x0.98                       0.000
channels_last cho conv 2D cua stereo                      87.6  x0.84                       0.514
fp16 autocast chi decoder                                 75.4  x0.98                       0.228
fp16 autocast chi trich dac trung                         75.5  x0.98                       0.858
GroupNorm song song (var_mean), chi decoder (6 lop)       50.5  x1.46                       0.015
GroupNorm song song, chi trich dac trung (17 lop)         65.4  x1.13                       0.082
GroupNorm song song, TAT CA 23 lop                        41.8  x1.77                       0.082
```

Mac dinh: 73.6 ms (script profile) va 73.8 ms (script GroupNorm). Moi cau hinh duoc huy va kiem tra duong mac dinh tro lai DUNG BIT truoc khi thu cau hinh tiep theo.

```
chat luong (8 frame val_7)            mac dinh  GroupNorm song song    chenh
------------------------------------  --------  -------------------  -------
EPE (px)                                3.3312               3.3313  +0.0000
left PSNR (dB)                         25.7342              25.7346  +0.0004
right PSNR (dB)                        23.1300              23.1306  +0.0006
lech disparity max / trung binh (px)         -       0.082 / 0.0008         
```

- [DO] Thay 23 lop GroupNorm bang phep tinh tuong duong dai so (var_mean + elementwise, cung tham so) giam 73.8 -> 41.8 ms (x1.77, 23.9 FPS) KHONG can train lai; EPE va PSNR gan nhu khong doi (chenh duoi 0.001 dB va 0.0001 px).
- [DO] Cac tuy chon runtime thong thuong (cudnn.benchmark, channels_last, fp16 cho tung phan) KHONG giup (x0.84-0.98). [SUY LUAN] Ly do co the la autocast giu group_norm o fp32 (hanh vi tai lieu cua PyTorch; chua kiem tra trong profile) nen phan nghen khong doi.
- [LUU Y] Hai dong 'autocast toan mang' trong lan chay profile bi VO HIEU (wrapper bao forward trong khi model goi predict_with_evidence nen khong co tac dung) - AMP toan mang CHUA do. Lan chay profile dau tien cua toi bi nhiem do loi channels_last (chuyen doi do dang) va da bi bo; chi dung lan hai. Ca hai file deu co trong thu muc bang chung.
- [LUU Y] Phien truoc (docs/kd_latency_frozen_actions_report_2026-10-28.md) tung thu cach nay (44 ms) nhung loai vi lech disparity 0.024 px vuot nguong 0.001 px dat truoc. Day la mot lua chon chinh sach ('tuong duong gan bit'), khong phai rang buoc chat luong: o muc metric thi khong thay khac biet.
- [SUY LUAN] Sau khi sua GroupNorm KD van cham hon baseline; so sanh ghep cung phien o muc 3.2 (tranh lech giua cac phien do may dung chung).
- [GIOI HAN] Chi 8 frame; phai chay lai benchmark 56 frame (EPE, PSNR, SSIM, LPIPS ca hai view) truoc khi dung ban sua cho moi tuyen bo. Toi CHUA sua ma nguon repo: moi thu nghiem chi nam trong bo nho; doi toan tu can ban dong y, test, va ghi 'xap xi so hoc' ben canh cong thuc chinh tac theo AGENTS.md.

### 3.2 Sua GroupNorm co tac dung voi muc tieu bai bao khong

```
do phan giai  baseline ms (FPS)  KD goc ms (FPS) [phien truoc]  KD da sua GN ms (FPS)  KD sua / baseline
------------  -----------------  -----------------------------  ---------------------  -----------------
512x640             30.3 (33.0)  76.6 (13.1)  [x2.33 baseline]            42.7 (23.4)  x1.41            
1024x1280           89.1 (11.2)  277.9 (3.6)  [x3.03 baseline]            107.6 (9.3)  x1.21            
```

So 'KD sua / baseline' la so sanh GHEP trong cung mot tien trinh (baseline va KD sua cung chay, cung phien). Baseline giua cac phien dao dong ~30-33 ms o 512x640 (31.95 benchmark chinh thuc), nen ty le 1.27-1.41 tuy phien; dung so ghep 1.41. Lan lap lai o 1024x1280: baseline 88.7 ms, KD sua 107.7 ms (on dinh). KD goc o cot 3 do o phien truoc (bang muc 3).

```
giai doan (KD da sua GN, 512x640)                      ms  % tong
---------------------------------------------------  ----  ------
mang stereo                                          15.8     37%
geometry prior (A03: khung be mat, hiep phuong sai)  12.8     30%
attribute head                                        1.1      3%
dung truong Gaussian (compose)                        3.6      8%
render (2 lan: trai va phai)                         12.8     30%
```

Thoi gian thuc (wall, median) 42.6 ms; cong cac giai doan 46.0 ms (khoang do su kien CUDA chong lan/ngan nhau chut). Khong con GroupNorm trong top.

- [DO] Sau khi sua GroupNorm, tong thoi gian kernel tren GPU chi 22.1 ms trong 42.6 ms thoi gian thuc (wall): GPU ranh ~48% thoi gian. Moi lan forward co ~3122 lan phong kernel va 145 lan dong bo host (cudaStreamSynchronize, ~23 ms CPU cho), gom ~142 lan goi .item()/bool(tensor).
- [SUY LUAN] Nghen khong con la tinh toan GPU ma la host: nhieu kernel nho (vong lap cost-volume theo 64 dich chuyen, geometry prior) va cac diem dong bo trong kiem tra hop le. Giam chung (bo dong bo trong duong deploy, hop nhat kernel, CUDA graph neu kha thi voi rasterizer) co the dua tong thoi gian ve gan tong kernel (~22 ms) - CHUA THU, day la can duoi tham khao chu khong phai du doan.
- [SUY LUAN] Co tac dung voi muc tieu bai bao khong? CO nhung CHUA DU. Sua GroupNorm (khong train lai, EPE/PSNR khong doi) la buoc quan trong nhat de luan diem toc do con ton tai: o do phan giai paper KD tu 3.6 FPS len 9.3 FPS (baseline 11.2 FPS, khop 11 cua paper), con cham 1.21 lan; o 512x640 con cham 1.41 lan. Nhung KD van CHUA nhanh hon baseline o ca hai do phan giai, va khi toi uu hoa ngang nhau cho ca hai thi khoang cach van con (muc 3.3).
- [SUY LUAN] Hai ve cua luan diem 'FPS cao, chat luong khong giam' so voi baseline khop: toc do (chua dat: 1.2-1.4 lan cham hon) va chat luong view trai (chua dat: left PSNR -1.4 dB, SSIM, LPIPS te hon; action hieu chinh nua trai chi them ~+0.85 dB view trai). Chat luong view phai va hinh hoc cua KD tot hon ro ret. Cach dat claim hien du lieu co the ung ho: 'toc do gan tuong duong (trong ~1.2-1.4 lan, co the thu hep them) nhung tong quat hoa qua sequence va chat luong view giu lai tot hon nhieu'; claim 'nhanh hon baseline' thi chua.
- [LUU Y] Toi uu hoa ky thuat (GroupNorm song song, bo dong bo) giup du lieu cua luan diem dung ra, nhung tu no KHONG la contribution khoa hoc; no la dieu kien can cho mot so sanh cong bang. Baseline cung chua duoc toi uu hoa tuong duong (no cung co dong bo host), nen khi cong bo can neu ro muc no luc cua moi ben.
- [LUU Y] O 1024x1280 ca hai mo hinh deu chi duoc train o 512x640 nen chat luong o do khong co y nghia (xem muc 3); so 1024x1280 chi la chi phi tinh toan, dung de doi chieu voi 11 FPS cua paper.

### 3.3 Con nhanh hon nua duoc khong, va khi toi uu hoa NGANG NHAU thi sao

```
cau hinh (512x640, batch 1, 8 frame)                            ms          FPS  so dong bo  giong bit            
-----------------------------------------------------  -----------  -----------  ----------  ---------------------
KD + GroupNorm song song (nhu muc 3.2)                        42.3         23.6         145  tham chieu           
  + bo kiem tra hop le gay dong bo                            36.5         27.4          73  co (disp+2 render)   
  + CUDA graph cho mang stereo                                33.1         30.2          73  co (disp+2 render)   
  (thay graph) torch.compile mang stereo                      29.7         33.7           -  KHONG                
baseline nguyen ban (do 2 lan: dau/cuoi)               30.1 / 29.9  33.2 / 33.5         123  tham chieu           
baseline + bo CUNG cac kiem tra (renderer dung chung)         22.8         43.8          51  co (disp+render trai)
```

'Giong bit' = disparity, render trai, render phai bang nhau tung bit voi tham chieu (eager chay hai lan cho ket qua giong nhau tung bit nen phep so sanh co nghia). torch.compile: lech disparity toi da 0.53 px, trung binh 0.0020 px; chua do o muc EPE/PSNR. Thoi gian bien dich 7 s.

```
ty le KD / baseline khi CUNG bo kiem tra dong bo  KD ms  baseline ms  ty le
------------------------------------------------  -----  -----------  -----
KD khong graph                                     36.5         22.8  x1.60
KD + CUDA graph (giong bit)                        33.1         22.8  x1.45
KD + torch.compile (khong giong bit)               29.7         22.8  x1.30
```

- [DO] Bo cac kiem tra hop le gay dong bo (finite, duong, trong [0,1]) o bo chuyen renderer dung chung lam KD nhanh x1.16 (42.3 -> 36.5 ms) nhung lam BASELINE nhanh hon: x1.32 (30.1 -> 22.8 ms), dau ra khong doi bit. Nghia la phan loi cua KD o muc 3.2 mot phan la loi chung cua ca hai.
- [SUY LUAN] Khi hai ben duoc toi uu hoa ngang nhau, KD van cham hon baseline o 512x640 khoang x1.30 den x1.60. Cam nhan 'KD gan ngang baseline' sau khi chi toi uu KD (muc 3.2) la ao giac do chi toi uu mot ben.
- [LUU Y] Baseline chua duoc CUDA-graph hoa phan encoder/RAFT, nen co the nhanh hon nua; vi vay cac ty le tren co the DANH GIA THAP khoang cach that. Toi chua do o 1024x1280 voi ma da toi uu.
- [LUU Y] Day la can tren chi mang tinh thu nghiem: bo han kiem tra la danh doi an toan (AGENTS.md cam nuot loi am tham). Phuong an hop le: kiem tra khong dong bo (tinh co tren thiet bi, doc mot lan moi frame hoac che do debug) - toi chua do chi phi cua phuong an nay. CUDA graph can hinh dang co dinh; torch.compile can kiem chung chat luong va cho 7-32 s bien dich.
- [DO] Con lai o KD: geometry prior ~12.8 ms GPU / 20.5 ms CPU, mang stereo 15.8 ms GPU, render 12.8 ms; ~73 dong bo con lai (nhieu nhat: disparity._count va composition._saturation_count, tuc cac phep dem de bao cao). Tong thoi gian kernel ~22 ms la can duoi cua cach tiep can hien tai - chua dat.
- [LUU Y] Lan chay dau cua script nay (spd_run1) co loi dung chung bo nho giua bo dem tinh va dau vao frame 0, lam nhieu cac phep so sanh bit; da chay lai (spd2) va chi dung ket qua lan hai. Ca hai file deu o thu muc bang chung.
- [LUU Y] Ban sao ma nguon da bo kiem tra chi nam trong thu muc tam (khong dung vao repo hay thu muc run); diff day du luu o results/EXPERIMENT_ONLY_nosync_validation.patch.

### 3.4 Chi toi uu phan cua minh (mang student), renderer dung chung va baseline giu nguyen

[DO] Day la cach so sanh ban de xuat: toi uu he thong cua minh, baseline nhu hien co. Renderer first-party dung chung KHONG bi sua (van co cac kiem tra dong bo) cho ca hai. Do ghep trong cung mot tien trinh, baseline do truoc va sau cac bien the KD de kiem tra troi; moi bien the KD duoc so bit voi KD eager cung do phan giai.

```
do phan giai  baseline nguyen ban ms (FPS)  KD + GN song song  + CUDA graph (giong bit)  + torch.compile (KHONG giong bit)
------------  ----------------------------  -----------------  ------------------------  ---------------------------------
512x640                        30.3 (33.0)  43.8 (22.8) x1.45         39.6 (25.2) x1.31                  37.0 (27.0) x1.22
1024x1280                      89.6 (11.2)  108.4 (9.2) x1.21         106.5 (9.4) x1.19                  92.5 (10.8) x1.03
```

Moi o KD: ms (FPS) x ty le so voi baseline (>1 nghia la KD cham hon). torch.compile: lech disparity toi da 512x640 0.53 px (tb 0.0020) / 1024x1280 3.39 px (tb 0.0046); chua do o muc EPE/PSNR. Thoi gian bien dich 27 s / 17 s.

```
baseline trong harness nay                                     512x640 ms (FPS)  1024x1280 ms (FPS)
-------------------------------------------------------------  ----------------  ------------------
nguyen ban (qua renderer first-party CO kiem tra)                   30.0 (33.4)         88.9 (11.2)
bo cac kiem tra dong bo do repo them (xap xi 'nhu phat hanh')       22.8 (43.8)         78.8 (12.7)
```

[DO] Dau ra cua baseline khong doi bit khi bo cac kiem tra o 512x640 (da kiem). Cot 'nhu phat hanh' chi la xap xi: toi CHUA do baseline qua duong render goc cua upstream.

- [DO] Chi toi uu mang student, KD khong vuot baseline (ty le ghep nhu bang tren): o 512x640 van cham x1.31 (graph) den x1.22 (compile); o 1024x1280 (do phan giai cua paper) cham x1.19 (graph) den x1.03 (compile) - tuc ~10.8 so voi ~11.2 FPS, gan ngang nhung KHONG nhanh hon.
- [LUU Y] Baseline 'nguyen ban' trong harness nay KHONG phai 'nhu phat hanh': no chay qua renderer first-party cua repo (co kiem tra dong bo ma upstream khong co). Bo cac kiem tra do baseline nhanh x1.31 o 512x640 va x1.13 o 1024x1280; neu dung xap xi 'nhu phat hanh', KD (compile) cham x1.62 o 512x640 va x1.17 o 1024x1280.
- [SUY LUAN] Muon noi 'nhanh hon' can toi uu them phan KHAC cua minh (geometry prior ~12.8 ms GPU / 20.5 ms CPU, cac dong bo con lai trong ma KD) - cung la 'phan cua minh', chua do. Cach trinh bay trung thuc cho bai bao: bao ca hai so (baseline qua duong render goc cua no va qua renderer first-party), noi ro cai gi duoc toi uu ben nao, va dung do phan giai paper (1024x1280) lam mat chinh vi o do KD gan ngang.
- [GIOI HAN] Chi 8 frame; torch.compile khong giong bit (cho phep kiem chung o muc metric 56 frame truoc khi dung); thoi gian chay tren may dung chung.

## 4. Vi sao action khong work

### 4.1 Ket qua Gate A (64 frame val_7 da niem phong)

```
action co dinh  dPSNR trai (dB)  CI95 theo keyframe  vi pham (known)  unknown  gate
--------------  ---------------  ------------------  ---------------  -------  ----
fixed stereo             -0.001  [-0.0016; -0.0002]              58%       27  fail
fixed gaussian           +0.004  [+0.0020; +0.0059]              19%       27  fail
```

[DO] Oracle (chon action tot nhat tung frame) chi tang left PSNR tu 25.1860 len 25.1904 dB (+0.0045); depth MAE 6.1651 -> 6.1614 mm. Khong selector nao vuot duoc oracle nay.

[XUNG DOT DA CONG BO] Code gate (stage_a_gate trong selector_evidence.py) them dieu kien 'khong co unknown' (27 unknown); dieu kien nay KHONG co trong van ban Gate A cua Plan 16 va Plan 18. Theo chu viet cua ke hoach, fixed Gaussian dat moi tieu chi (CI duong, 4/4 keyframe duong, vi pham 19.4% <= 25%). Khong gan nhan pass; ket luan khoa hoc khong doi vi muc tang chi +0.004 dB.

### 4.2 Tran theo do phu (tu log train, 1000 mau cuoi)

- [DO] Moi can thiep ghi 1 loi 32x32 = 1,024 px = 0.31% khung 512x640 (4 patch/frame = 1.25%).
- [DO] Sua HOAN HAO mot loi chi giam toi da ~0.29% loss RGB toan khung (~0.01-0.03 dB). Mau so gia dinh toan khung; mau so that la vung support, chua do, nen tran co the cao hon chut.
- [DO] Gaussian repair giam 19.2% loss cuc bo (thang 90% patch, hai 3.6%) nhung chi 0.080% loss toan khung; stereo giam 8.5% cuc bo (thang 66%, hai 21.1%), 0.054% toan khung.
- [DO] Loi trai deu: loss cuc bo / toan khung = 0.93, khong co diem nong de va.
- [DO] Muc tieu train la loss cuc bo tren patch, con gate do PSNR toan khung -> lech muc tieu.
- [DO] GT depth hop le chiem trung vi 11.9% khung o dev (0-28%), 19.5% o train; trong 4 patch cua 1 frame trung vi co 836/4096 px GT, 7% frame train khong co px nao. (Lan bao dau toi noi ~5% la SAI, xem muc 8.)

### 4.3 Thi nghiem do phu (24 frame val_7 ngoai nhom 64 niem phong va 56 benchmark)

```
vung ghi                  dPSNR T  dPSNR P  frame T tot  frame P tot
------------------------  -------  -------  -----------  -----------
1 loi 32x32 (nhu Gate A)    -0.00    +0.00        22/24        20/24
4 loi 32x32                 +0.01    +0.01        22/24        21/24
tile 0 (tren trai)          +0.17    +0.09        18/24        20/24
tile 1 (tren phai)          +0.37    -0.52        24/24         3/24
tile 2 (duoi trai)          +0.66    +0.34        24/24        23/24
tile 3 (duoi phai)          +1.17    -1.48        23/24         1/24
TOAN KHUNG                  +2.97    -1.61        24/24         2/24
```

T = view trai (left), P = view phai (right); dPSNR tinh bang dB so voi STOP; frame T/P tot = so frame co PSNR cao hon STOP. Voi loi 32x32 chenh lech < 0.01 dB nen viec dem 'frame tot hon' o hai dong dau khong co y nghia (nhieu). Cac dong con lai moi la tin hieu.

[DO] Tong 4 tile chay rieng: trai +2.37, phai -1.57 dB; toan khung: trai +2.97, phai -1.61 dB -> gan cong tinh (~80% o view trai, ~98% o view phai). Chi phi action ~17-18 ms bat ke kich thuoc vung (luon render lai ca hai view).

```
policy              dPSNR T  dPSNR P  dSSIM T  dEPE px (am=tot)  frame EPE tot
------------------  -------  -------  -------  ----------------  -------------
stereo, 1 tile        -0.05    +0.01  -0.0057            -0.104          11/23
stereo, toan khung    -0.43    +0.11  -0.0305            -0.131          11/23
```

[DO] Stereo repair phu rong lam left PSNR te hon (toan khung: 20/24 frame te hon, left SSIM te hon 24/24) va EPE khong nhat quan. Ket luan: bo stereo repair.

### 4.4 Co che: action thuc su lam gi

- [DO] Opacity-logit bi day sat bien +2.0 o 98% primitive; opacity trung vi 0.99926 -> 0.99990.
- [DO] Scale: truc x,y tang log +0.42/+0.40 (~x1.5); truc z (phap tuyen) -0.75 = dung bien -0.75 (~x0.47). Khong scale nao cham san (0%). Scale trung vi 9.6e-05 -> 1.5e-04 m. Day KHONG phai su sup scale 1e-38 cua baseline.
- [SUY LUAN] Action la hieu chinh he thong bi gioi han boi bien (z va opacity dung bien), khong phai sua cuc bo theo ngu canh.

```
bien the (24 frame)        dPSNR T  dPSNR P  dSSIM T  dSSIM P  TB 2 view
-------------------------  -------  -------  -------  -------  ---------
action da hoc, toan khung    +2.97    -1.61   +0.042   +0.016      +0.68
HANG SO = TB cua action      +1.98    -1.44   +0.044   +0.014      +0.27
chi ep z x0.47               +1.34    -1.95   +0.055   -0.018      -0.30
chi ep z x0.7                +1.19    -0.69   +0.031   -0.000      +0.25
chi phinh x,y x1.5           -0.78    -0.14   -0.018   +0.003      -0.46
chi opacity +2               +0.00    +0.00   -0.000   -0.000      +0.00
x,y x1.25 + z x0.47          +2.22    -1.54   +0.056   +0.007      +0.34
```

HANG SO = nhan scale x1.52, x1.49, x0.47 cho 3 truc va cong +2.0 vao opacity-logit cho MOI primitive (khong mang). Luu y: 'TB hai view' la tom tat toi chon SAU khi thay du lieu, khong phai tieu chi thanh cong. Luoi tham so nay thua (z 0.47 va 0.7; chua thu 0.7-1.0).

[DO] Do lech chuan cua dlogscale giua cac primitive (trung binh frame): x 0.30, y 0.39, z 0.017; opacity-logit 0.032. Tuc la z va opacity gan nhu HANG SO; chi phan phinh trong mat phang (q10..q90 ~ -0.02..+0.70) phu thuoc dau vao, gop them ~+1 dB view trai so voi ban hang so.

[DO] Lech mau (view phai toi hon) chi giai thich ~11% muc tang MSE view phai; phan con lai khong phai do sang. [GIA THUYET] Phinh primitive trong mat phang lam nhoe o view moi khi do sau sai (EPE ~3.4 px); view trai thi vi tri chinh xac theo cau truc. Chua kiem chung. Mot gia thuyet khac chua loai tru: lo hong phu (coverage holes) - nen so sanh alpha tich luy cua view phai truoc/sau.

### 4.5 Quy luat theo vi tri: nua trai / nua phai (chay truc tiep)

```
tap                          vung        dPSNR T  dPSNR P  frame P tot  frame CA HAI tot
---------------------------  ----------  -------  -------  -----------  ----------------
val_7 (24 frame)             nua TRAI      +0.85    +0.44        24/24             24/24
val_7 (24 frame)             nua PHAI      +1.65    -1.88         0/24              0/24
val_7 (24 frame)             toan khung    +2.97    -1.61         2/24              2/24
train (16 frame, in-sample)  nua TRAI      +0.84    +0.75        16/16             16/16
train (16 frame, in-sample)  nua PHAI      +0.53    -1.89         0/16              0/16
train (16 frame, in-sample)  toan khung    +1.52    -1.38         3/16              3/16
```

in-sample = frame train ma trong so action da thay luc huan luyen (khong doc lap).

```
dataset train  nua trai: T  nua trai: P  nua phai: P
-------------  -----------  -----------  -----------
dataset_1            +0.84        +0.70        -1.24
dataset_2            +1.50        +1.17        -0.82
dataset_3            +0.78        +0.73        -3.97
dataset_6            +0.23        +0.38        -1.53
```

[DO] Quy luat nhat quan tren 5 dataset (7, 1, 2, 3, 6): nua trai cai thien ca hai view, nua phai lam hai view phai. Co che chua ro [GIA THUYET: hinh hoc stereo / vung nhin thay]. [SUY LUAN] Chon tap tile tot nhat tung frame (cong gan dung, toi uu vi chon va do tren cung frame) cho TB hai view +0.91 dB, quy tac co dinh 'nua trai' cho +0.65 dB -> headroom cua routing ~+0.27 dB: gan nhu khong dang ke, mot quy tac co dinh la du.

## 5. Baseline: khong hong, nhung khong tong quat qua sequence

```
tap                      n frame  EPE TB (px)  EPE trung vi (px)
-----------------------  -------  -----------  -----------------
train (dataset 1,2,3,6)        8         0.59               0.56
val_7                          8        14.21              11.84
```

CPU FP32, chi duong stereo, cung cong thuc EPE (GT hop le & support & huu han). Val_7 tai hien ~15.5 px cua benchmark.

- [DO] Tren train, 0-8% pixel sai >3 px; tren val_7, 41-100%. GT disparity train 26-41 px, val_7 22-80 px; du doan cua baseline bi keo ve ~38-63 px.
- [SUY LUAN] KD doc cung du lieu val_7 qua cung adapter va dat ~3.4-3.6 px, nen duong du lieu khong loi; loi the hinh hoc cua KD la that trong protocol khop nay (train 4 sequence, kiem tra sequence khac).
- [LUU Y] Day khong phai protocol trong-sequence cua paper, nen khong noi gi ve so lieu cua paper. Cach dien giai trung thuc: giao vien nen tang tong quat tot hon mang train bang GT bi overfit.
- [DO] Diem view TRAI cua baseline khong phan anh chat luong 3D: no khong doi khi lam sai do sau (muc 5.1). Day la bang chung truc tiep cho nghi van sup scale / sao chep anh nguon, thay cho gia thuyet truoc.

```
chi so (benchmark 56 frame val_7)  baseline      KD
---------------------------------  --------  ------
disparity EPE (px)                    15.47    3.62
depth MAE (mm)                        18.80    4.37
left PSNR (dB)                        26.89   25.44
right PSNR (dB)                       17.33   22.10
left SSIM                            0.9375  0.7885
left LPIPS                           0.0484  0.2628
```

Nguon: docs/baseline_stage2_gpu0_restart_2026-10-28.md (muc 'Verified completion').

### 5.1 Chat luong cua KD co hon baseline khong

[DO] Thu nghiem co kiem soat (8 frame val_7, 512x640): lam sai CO Y do sau ngay sau mang stereo (nhan 0.7, nhan 1.3, cong 8 px, hoac thay bang mat phang phang = trung binh khung) roi do lai ca hai view. Neu mot view phan anh dung chat luong 3D thi no phai te di khi do sau sai.

```
model     lam sai do sau           L PSNR  R PSNR  L SSIM  R SSIM  EPE px
--------  -----------------------  ------  ------  ------  ------  ------
baseline  none                      26.89   17.94   0.938   0.584    14.2
baseline  disparity x0.7            26.89   17.54   0.938   0.567    19.6
baseline  disparity x1.3            26.89   17.20   0.938   0.528    18.5
baseline  disparity +8 px           26.89   17.66   0.938   0.566    15.6
baseline  flat plane (frame mean)   27.25   17.78   0.975   0.593    16.8
kd        none                      25.73   23.13   0.783   0.732     3.3
kd        disparity x0.7            27.19   20.48   0.816   0.615    14.2
kd        disparity x1.3            24.56   18.72   0.753   0.605    17.5
kd        disparity +8 px           25.54   20.57   0.779   0.627    10.0
kd        flat plane (frame mean)   26.58   17.82   0.924   0.571    17.8
```

- [DO] Baseline: left PSNR 26.89 va SSIM 0.938 KHONG DOI khi do sau bi nhan 0.7, nhan 1.3 hoac cong 8 px (EPE tang 14 -> 19 px), va con CAO HON voi mat phang phang (27.25 / 0.975). [SUY LUAN] Co che co the la moi diem Gaussian nam tren tia cua chinh pixel no va co kich thuoc ~0 (scale ~1e-38 da do o muc 3), nen view nguon tai tao anh dau vao bat ke do sau. Dieu chac chan (da do) la diem view trai cua baseline khong phu thuoc do chinh xac cua do sau, nen khong do duoc chat luong tai tao 3D.
- [DO] Baseline view phai (17.94 dB) gan bang mat phang phang (17.78 dB): do sau baseline du doan tren val_7 gan nhu khong giup gi cho view moi. KD view phai 23.13 dB so voi mat phang phang 17.82 dB (+5.3 dB): do sau cua KD mang thong tin that.
- [DO] KD view trai CO nhay voi do sau/kich thuoc Gaussian va co the bi 'mua' bang cach lam sai: do sau x0.7 tang left PSNR 25.73 -> 27.19 dB nhung giam right PSNR 23.13 -> 20.48 dB. [SUY LUAN] Co the phan thua view trai cua KD so voi baseline lien quan toi kich thuoc/mo cua Gaussian (cung chieu voi ket qua hieu chinh Gaussian o muc 4) chu khong phai do hinh hoc; chua kiem chung truc tiep.
- [SUY LUAN] Ket luan trung thuc ve 'chat luong': KD HON ro ret o nhung gi do duoc chat luong 3D that (hinh hoc EPE 3.6 vs 15.5 px, depth MAE 4.4 vs 18.8 mm; view giu lai right PSNR 22.1 vs 17.3 dB, right SSIM 0.726 vs 0.553; right LPIPS xap xi 0.317 vs 0.321). Baseline CAO HON o view trai (PSNR 26.9 vs 25.4, SSIM 0.94 vs 0.79, LPIPS 0.048 vs 0.263) nhung diem nay khong do duoc chat luong 3D. Khong the noi 'KD hon moi mat' hay 'KD khong kem baseline ve view trai'.
- [GIOI HAN] Thu nghiem lam sai do sau chi 8 frame; so benchmark chinh thuc 56 frame, mot dataset (val_7), mot seed moi model. Baseline duoc train bang GT tren 4 sequence va khong tong quat sang sequence khac: day la bang chung ve tong quat hoa qua sequence trong protocol khop nay, khong phai ve moi dieu kien (protocol trong-sequence cua paper se co loi cho baseline). Viec sao chep anh nguon la dac diem cua mo hinh baseline nay (scale ~1e-38), chua kiem tra o checkpoint khac.

## 6. Danh gia contribution va khuyen nghi

- [SUY LUAN] Action chay luc suy luan khong phai contribution chinh: tran thuc te ~+1 dB TB hai view, them ~18 ms, phan lon la hieu chinh do day Gaussian cua head student (hang so), khong cham toi khoang cach toc do.
- [SUY LUAN] Cai bo phan ung ho duoc bang du lieu: (a) tong quat hoa qua sequence cua student chung cat tu teacher nen tang; (b) chat luong view giu lai tot hon baseline.
- [SUY LUAN] Luan diem 'FPS cao + chat luong khong giam' hien chua duoc ung ho: xem muc 3. Neu muon giu no thi phai thiet ke lai student (decoder GroupNorm) va train lai.
- Khuyen nghi: bo stereo repair va selector/routing; khong tiep tuc sua action patch; neu muon giu hieu chinh thi dua vao head thuoc tinh cua student (fine-tune bang render loss toan khung, cham ca hai view, khong them latency) - can ban duyet.

## 7. Quyet dinh can ban dua ra (toi khong tu quyet)

- Huong paper: (a) tong quat hoa + chat luong hai view, du lieu hien da ung ho phan lon; hay (b) giu luan diem toc do (phai thiet ke lai student, ket qua khong chac).
- Co duyet fine-tune head thuoc tinh cua student bang render loss toan khung khong; hay chi giu nhu mot dong ablation?
- Neu muon xac nhan quy tac 'nua trai' tren 64 frame da niem phong: ban dat truoc nguong cho CA HAI view (nhat la muc giam toi da chap nhan o view phai) va tieu chi chinh.
- Kiem tra hop le trong duong suy luan (finite/duong/[0,1]) dang gay ~70 lan dong bo host cho CA HAI mo hinh: co chuyen sang kiem tra khong dong bo (hoac che do debug) khong? Day la thay doi an toan, cua ban.
- Co chap nhan tuong duong o muc metric (thay vi 0.001 px) de dung GroupNorm song song khong? Neu co: toi se sua o module chuan, them test va chay lai benchmark 56 frame truoc; day la dieu kien can cho claim toc do nhung chua dam bao KD nhanh hon baseline (muc 3.2).
- Co bo luan diem routing chon loc (RiskRoute) khong? Du lieu cho thay headroom ~0.3 dB so voi quy tac co dinh.

## 8. Do tin cay, gioi han, va cac loi toi da sua

- Moi thi nghiem muc 4.3-4.5 la tham do: 24 frame val_7 (ngoai nhom 64 va 56) + 16 frame train; chon va do tren cung frame nen so lac quan; chua pre-register; train la in-sample voi trong so action.
- Cong gan dung theo tile gia dinh cong tinh; nua trai/phai da duoc chay truc tiep va khop (+0.85/+0.44 vs +0.83/+0.44).
- Latency do tren may dung chung (job GPU1 cua ban chay), WSL, khong phai may chuyen dung; khong tinh I/O. KD chi duoc train o 512x640: latency o 1024x1280 do duoc, chat luong o do phan giai do khong co y nghia khoa hoc neu EPE/PSNR te.
- Cac loi toi da noi sai va da sua trong phien: (1) GT depth '~5%' -> thuc te trung vi 11.9% dev / 19.5% train; (2) render chiem '10-20%' thoi gian -> ~38% o 512x640, ~21% o 1280x1024; (3) 'sieu tuyen tinh 18 lan' -> sai, do so tile yeu nhat voi toan khung; (4) 'router khong co viec' -> headroom ~0.3 dB; (5) thoi gian action 711 ms/1398 ms o lan chay dau la artifact cua empty_cache moi frame, da bo; (6) tran 0.29% gia dinh mau so toan khung.
- Su co ha tang: phep do GPU thu hai bi treo trong driver WSL (dxgprocess_close_adapter) va duoc giai phong sau do; cac lan chay sau hoan thanh binh thuong.

## 9. Nguon goc du lieu va cach tai chay

```
file (trong thu muc bang chung)                               sha256 (16 ky tu dau)
------------------------------------------------------------  ---------------------
lat_interpretation.txt                                        dd414d1a680a364d     
results/EXPERIMENT_ONLY_nosync_validation.patch               b00b8416d4d77e48     
results/bf1280_fork.json                                      1b4cb6eb0d3bfaa4     
results/bf1280_orig.json                                      57882c1554b004b8     
results/bf1280_orig2.json                                     ae6afcb61ef4fb84     
results/bfair_fork.json                                       98e92077efbe7668     
results/bfair_orig.json                                       4dcb06c64b56faa6     
results/bfair_orig2.json                                      5e0c893c0172cb74     
results/cov_diag2.json                                        2c95c20c02811721     
results/epe_cpu.json                                          265bc3c069a825f0     
results/fps_diag.json                                         2db36e5e6aebb61a     
results/gnp.json                                              11b862dec640ffbb     
results/halves.json                                           99806a5705c4fad9     
results/kdprof2.json                                          833230268ebacc9b     
results/kdprof3.json                                          bca3ed6ce2345cf0     
results/kdprof3_run1_FAILED_partial.log                       b1a4d775a77a109e     
results/kdprof_run1_CONTAMINATED_by_channels_last_error.json  99e7673db144a5bc     
results/lat.json                                              7154c60889e4fcb4     
results/latfix.json                                           8ecf5ea07b5e6b29     
results/ldi.json                                              ae4e93ca36e733cc     
results/ourpart.json                                          632db0831219b2a6     
results/scan.json                                             282ae4c0fb3cc259     
results/spd2_fast.json                                        60744f573e73a2ca     
results/spd2_ref.json                                         163f181f2f17a5c2     
results/spd_run1_CONTAMINATED_static_buffer_aliasing.json     912b9a8db59507b7     
results/train_action_gain_by_1000_samples.txt                 fbc0fea1a3efd602     
results/train_curve_by_1000_samples.txt                       4cd3b894613c2032     
results/train_last1000_patch_stats.txt                        bacd47b2f86ea4e6     
scripts/action_coverage_diag.py                               e648212bc53bd86f     
scripts/action_coverage_diag2.py                              7a312d3dfd741f02     
scripts/action_halves.py                                      2036b5d35e8c381f     
scripts/action_scan.py                                        f8de1b0bb50143d3     
scripts/baseline_epe_cpu.py                                   e4a55fa413e81cd6     
scripts/baseline_fps_diag.py                                  f4d3716cfcedbd75     
scripts/baseline_fps_diag2.py                                 1a861848a14dd29e     
scripts/baseline_sync_fair.py                                 03c5ad58636f4e11     
scripts/baseline_sync_fair_1280.py                            064168ddc6ccf27a     
scripts/ckpt_peek.py                                          000ecfb919357188     
scripts/curve.py                                              32ed9b572065b064     
scripts/envsrc.sh                                             c73cce4dd06edde0     
scripts/gains.py                                              47cab30f9fead1e2     
scripts/gn_patch.py                                           9209b36dc76e4a59     
scripts/gpu_probe.py                                          86ed674920a1e8ba     
scripts/kd_ourpart.py                                         7b7c8742199c17e6     
scripts/kd_profile.py                                         66185ecb77742b17     
scripts/kd_profile2.py                                        738c8c7505753efa     
scripts/kd_profile3.py                                        d146727f25ec31a0     
scripts/kd_speedups.py                                        94cb2d7ac2ef2ff2     
scripts/kd_speedups2.py                                       537e4b5b0594995a     
scripts/latency_gnfix.py                                      6eb079cf7818f52d     
scripts/latency_matched.py                                    fff053c56a2f8a3e     
scripts/left_view_depth_independence.py                       85208752259d8464     
scripts/make_report.py                                        b302853ddf88b92f     
scripts/patchstats.py                                         bd61babdfb012630     
```

Thu muc bang chung: /home/bkai/.local/share/reliableendo-gs/reports/phase2-action-latency-diagnosis-20261029. Moi script ghi vao thu muc rieng, khong ghi vao run, checkpoint hay repo. Run goc: /home/bkai/.local/share/reliableendo-gs/runs/phase2-frozen-actions-20261028T114155Z.

- Checkpoint Stage A: stage-a-attempt-00/final.pt sha256 ef7fa35f49e1ce21bd5994dedcb93b638750f98ca21afa68f37d1feb24fb4ba4.
- KD final.pt sha256 f77547ef2cdc3d5030d537d83a203c0d9f8cf3978020c62a370af78d36cedd62; baseline ckpt-s2-010000.pt sha256 7c48dcc83994a4622d925b8b47c6c877d2db117054647a23a931979595f1106b.
- Source Stage A trung khit working tree (src va configs) tai thoi diem phan tich; workspace Git HEAD ac97de8 va CON thay doi chua commit (khong phai mot lan chay Git sach).
- Tai chay: moi script nhan duong dan dau ra va log lam doi so; chay bang envrun.sh cua run (ghim GPU0 theo UUID). Dat gioi han thoi gian cung va luon ghi log ra file rieng, khong qua pipe.

## 10. Cach doc report nay tren terminal bi loi hien thi

Hai ban cung noi dung: .txt (ASCII thuan, khong dau, xuong dong o cot 96) va .md. Dung ban .txt neu terminal loi font/dau. Thu muc: /mnt/d/workspace/huann/reliableendo-gs.

```
cd /mnt/d/workspace/huann/reliableendo-gs
F=docs/phase2_action_latency_diagnosis_2026-10-29.txt

less -S $F                 # xem tung trang; mui ten/PgUp/PgDn; /tu_khoa de tim; q de thoat
less -N $F                 # kem so dong
grep -n '^[0-9]*\. ' $F    # muc luc: so dong cua tung muc 1..10
awk '/^3\. /,/^4\. /' $F     # doc muc 3 (latency)
awk '/^3\.1 /,/^4\. /' $F   # doc muc 3.1: vi sao student cham
awk '/^3\.2 /,/^4\. /' $F   # doc muc 3.2: tac dung voi muc tieu bai bao
awk '/^3\.3 /,/^4\. /' $F   # doc muc 3.3 va 3.4: toi uu them, so sanh cong bang
awk '/^5\. /,/^6\. /' $F   # doc muc 5 va 5.1: chat luong KD so voi baseline
awk '/^4\. /,/^5\. /' $F     # doc muc 4 (action); doi so de doc muc khac
head -60 $F                # chi doc phan dau (tom tat)
wc -l $F                   # tong so dong

# neu terminal van loi hien thi:
reset; stty sane           # khoi phuc terminal
echo $TERM; locale         # kiem tra; thu: export LANG=C.UTF-8
notepad.exe "$(wslpath -w $F)"      # mo bang Notepad cua Windows
code $F                    # mo bang VS Code (neu co)
```
