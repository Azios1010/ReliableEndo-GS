# Baseline feasibility addendum — 2026-10-31

Task C is a **source survey**, not a benchmark run. This new document supplements
[the original feasibility report](baseline_feasibility_2026-10-31.md) and
[the run/preparation report](baseline_run_report_2026-10-31.md); neither was edited.
Availability was checked on 2026-10-31. Engineering estimates are judgments, not
measured runtimes or promises that a model fits the GPU limit.

The most useful additions are geometry baselines and a qualified surgical
Splatt3R derivative, **Endo-PairGS**. Endo3R is released, but its documented training
includes dataset 7. EndoVGGT is withdrawn. GPS-Gaussian advertises weights, but an
accessible, locally hashed official checkpoint remains unverified. No additional
complete, released, single-stereo-pair endoscopic Gaussian renderer satisfying our
protocol was verified in the sources checked. That is a bounded search finding,
not a claim that Endo-E2E-GS is the only possible method.

## Scope and legitimate comparison axes

No model inference, training, GPU work, environment installation or project-code
change was performed for Task C. No local dataset images, sealed-64 worklist,
48-frame development cohort, or final-test manifests/data were opened. Public
upstream **training-list source files** were inspected to identify potential
training overlap; this does not evaluate or access our reserved data. EndoNeRF
data were not downloaded and its public data folder was not inspected.

Every negative availability finding below means **“not verified in the sources
checked.”** “Weights linked” means an official source advertises them, not that
their bytes have been downloaded, loaded or verified offline. A paper license
does not establish a code or checkpoint license. Exact inspected commits are in
the commit register below; where no official repository was verified, the commit
is N/A.

| Axis | Eligible outputs and metrics | Conditions for our protocol |
| --- | --- | --- |
| Native reconstruction/rendering | Left and right PSNR, SSIM and LPIPS from a method's own Gaussian/field renderer | Render the two specified A00 cameras at 512×640, using only the allowed input pair. Disclose methods that use both images' colours: reconstruction of an input camera is not held-out novel-view synthesis. Preserve the frozen metric/mask definitions. |
| Stereo geometry | Native disparity EPE and canonical stereo depth MAE/RMSE | Disparity must be `u_left - u_right` in A00 input pixels; include principal-point offsets using the repository's canonical conversion. Verify resizing, sign and optical-depth units independently of quality. No GT scale fitting. |
| Monocular geometry | Depth MAE/RMSE only if raw output has a justified conversion to metres; optionally **derived** disparity EPE | Report the weaker/different monocular input condition. Scale consistency across frames is not absolute metric calibration. Derive disparity only after optical-z, intrinsics and units are established, and label it derived. No native RGB-render metrics for depth-only networks. |
| Temporal reconstruction | Geometry, or native rendering if provided, in a separate temporal-input group | A recurrent model receiving extra frames has additional information. The 56 sparse val samples do not authorize reading their neighbours; context policy needs separate authorization. Resetting memory per sample is a disclosed recipe change, not equivalent native video inference. |
| Scene fitting / aligned references | Reference table only | Training/test frames from one sequence, bundle/global alignment, deformation fitting, or GT scale alignment are different protocols. GT-aligned references would **favour the baseline** and are excluded from the requested raw metric table; none were run. |

Continue to use the fixed final/released checkpoint rule, never val_7 quality to
choose a model, size, scale, crop, bounds or checkpoint. If training were later
authorized, use train_1236 only, final checkpoint only, and disclose unmatched
compute. No frozen metric implementation needs changing for this survey. A
depth-only method must not acquire “rendering scores” through an unreported new
Gaussian head or custom image warp.

## Requested candidates

Abbreviations: **R** = native left/right PSNR, SSIM, LPIPS; **G** = native disparity
EPE and depth MAE; **Z** = raw metric depth MAE, with derived EPE only after the
conditions above. Conditional entries are not declarations of runnable models.

| Candidate / verified primary source | Licence; code and weights status | Inputs → outputs | Axis, protocol fit, effort and recommendation |
| --- | --- | --- | --- |
| **Endo3R**, [arXiv:2504.03198](https://arxiv.org/abs/2504.03198), [official repo](https://github.com/wrld/Endo3R) | Root MIT; included DUSt3R material retains CC BY-NC-SA 4.0 notices. Inference/evaluation code and official Drive checkpoint linked; training-code release is still a README TODO. Weight bytes not downloaded. | Dynamic monocular video and recurrent memory → global point maps, depth, camera estimates; no native Gaussian renderer or stereo-disparity head. | **Z conditional**, no R. Documented SCARED training 1–7 includes our val_7 and excluded 4/5. Native depth evaluation fits one GT median scale across the sequence. About 1–2 days to bridge depth, but **exclude as an unseen-val_7 competitor** and do not use aligned paper scores. |
| **EndoVGGT**, [arXiv:2603.24577 and history](https://arxiv.org/abs/2603.24577) | Current v2 is withdrawn; no license for that version. Official runnable code/weights not verified in the sources checked. | Claimed monocular reconstruction/depth framework; no executable interface verified. | **Exclude.** Do not import reported metrics. Reimplementation is outside the three-day limit. |
| **StereoNeRF**, *Generalizable Novel-View Synthesis using a Stereo Camera*, [arXiv:2404.13541](https://arxiv.org/abs/2404.13541), [official repo](https://github.com/Haechan21/StereoNeRF) | No root code LICENSE found; GeoNeRF-derived source carries GPLv3 notices. README's CC BY-NC 3.0 section licenses the **datasets**, not all code. Inference code and official Drive weights linked; UniMatch weights and precomputed disparities also needed. | Calibrated views from **multiple stereo-camera positions**, K/extrinsics, near/far and disparity preprocessing → radiance field, rendered RGB and depth; not Gaussians. | R/Z in principle, but released config uses **three stereo pairs**, not our one pair. One-pair operation and 512×640 within 3.2 GiB are not verified. **Defer**: likely exceeds three days if architecture/dependency changes are needed. |
| **Splatt3R**, [arXiv:2408.13912](https://arxiv.org/abs/2408.13912), [official repo](https://github.com/btsmart/splatt3r) | Root CC BY-NC 4.0, MASt3R CC BY-NC-SA 4.0, and separate Graphdeco rasterizer terms. Official HF checkpoint present in model metadata, with advertised LFS hash below; not downloaded. | Two uncalibrated RGB views → point maps and Gaussians in the first-camera frame, native RGB renders. No native disparity head. | R/Z **conditional** on scale/frame and uncropped grid validation. Default demo makes **512×512 square crops**, not full 512×640 A00. Offline operation is possible in principle with all files local, but not tested. About 2–3 days; reserve option, not ahead of prepared posed baselines. |
| **GPS-Gaussian**, [arXiv:2312.02155](https://arxiv.org/abs/2312.02155), [official repo](https://github.com/aipixel/GPS-Gaussian) | MIT root plus separate rasterizer terms. Code verified. README links `GPS-GS_stage2_final.pth` via Baidu/OneDrive; OneDrive returned **403** in this check. No official weight bytes/local SHA-256 verified. | Rectified RGB pair, masks, intrinsics/extrinsics and stereo geometry → flow/disparity-related depth, pixel Gaussians and native RGB renders. | R/G conditional on weights, flow sign/units and human-domain preprocessing. Rectified pair is appropriate; full 512×640 and 20% memory cap untested. About 1–2 days **after** weight access; **blocked by retrieval**, not disproven as a method. |
| **StereoSplat — Yugay et al.**, [arXiv:2610.35902](https://arxiv.org/abs/2610.35902), [author project](https://vladimiryugay.github.io/stereosplat/) | Fresh project-page retrieval still says **Coming Soon** and links the paper. Runnable official code, weights and their licences not verified. | Calibrated stereo pair → metric geometry/Gaussians and rendered views, as described by paper. | Potential R/G, **not runnable verified**. Cite only; reimplementation exceeds three days. Do not confuse with Liu/Okutomi's same-name work in the original report. |
| **StereoGaussians**, [arXiv:2609.38592](https://arxiv.org/abs/2609.38592), [v1 full text](https://arxiv.org/html/2609.38592v1) | Paper v1 verified; official executable code, weights and code licence not verified in paper/project searches. | Calibrated stereo pair; frozen stereo features → metric, layered expanded-canvas Gaussians and RGB. | Potential R/G, but **cite only** without release. Teacher-scene preparation and retraining exceed three days. |

### Endo3R: training overlap and scale are distinct obstacles

The [pinned README](https://github.com/wrld/Endo3R/blob/8145c8e5535d2111974561e7d1cc06f1d008fe5e/README.md)
explicitly lists SCARED **datasets 1–7 for training and 8–9 for evaluation**, alongside
StereoMIS, C3VD and EndoMapper, plus additional unlabeled video. It describes
StereoAnything pseudo-depth in training. Thus the released model is neither
trained on our restricted train_1236 nor a clean unseen-sequence comparison on
dataset 7. This finding comes from documentation, not from opening any of our
reserved datasets.

The [depth evaluation implementation](https://github.com/wrld/Endo3R/blob/8145c8e5535d2111974561e7d1cc06f1d008fe5e/eval/depth_evaluation.py)
concatenates valid predicted/GT depths, computes a single median ratio and scales
predictions before scoring. That is **sequence-level GT scale alignment**, not
per-frame alignment, and is incompatible with our raw metric protocol. A fair
future geometry experiment would need documented raw depth units, optical-z and
image-grid conversion, a declared monocular/temporal input policy, and no GT scale
fit. The README's native evaluation uses 320×256. Neither the word “scale-consistent”
nor pseudo-depth supervision proves that the downloaded model would output metres
on our rig. If scale remains ambiguous, record the question and stop; do not
invent a stereo scale for a monocular network.

### EndoVGGT: confirmed withdrawal

The arXiv listing records **v1: 2026-03-25 17:53:34 UTC** and **v2:
2026-05-11 23:16:46 UTC, withdrawn**. The current page names the withdrawing author,
offers no current PDF, and explains that presentation, methodology description and
results analysis need substantial correction. The retained v1 does not override
this status. No result from this work is used here. This is a withdrawal finding;
the separate claim that an official release exists is **not verified in the
sources checked**. [Primary listing and version history](https://arxiv.org/abs/2603.24577).

### StereoNeRF: calibrated stereo does not mean one stereo pair

At the pinned commit,
[`config_general_stereo_real.txt`](https://github.com/Haechan21/StereoNeRF/blob/c7fe2fc10ca57ed7a86e113f29e0b699507b09da/configs/config_general_stereo_real.txt)
sets `nb_views=6`, ray batch 512 and chunk 4096. The
[dataset factory](https://github.com/Haechan21/StereoNeRF/blob/c7fe2fc10ca57ed7a86e113f29e0b699507b09da/data/get_datasets.py)
uses `img_wh=(512,256)`; the stereo loader halves the view count into stereo
positions. This is three pairs at H=256, W=512. Our H=512, W=640 grid has 2.5 times
as many pixels, before changing the view count. Reducing ray chunks can help
rendering memory, but does not establish feature/cost-volume memory feasibility.

The [official instructions](https://github.com/Haechan21/StereoNeRF/blob/c7fe2fc10ca57ed7a86e113f29e0b699507b09da/README.md)
specify PyTorch 1.9.1/CUDA 11.1 and disparity preprocessing with UniMatch. The
[author supplement](https://jinwonjoon.github.io/stereonerf/StereoNeRF_files/supp.pdf)
reports five training days on an RTX 3090; that is author training cost, not a local
port measurement. No retraining is proposed. A one-pair run would require a
documented, compatible source-view configuration, not simply dropping four views
from the benchmark without disclosure.

### Splatt3R and GPS: offline, grid and licence checks

Splatt3R's [demo](https://github.com/btsmart/splatt3r/blob/bda1dd07cf84b9189baabb1b8beac243959cfe5d/demo.py)
retrieves its checkpoint through HF. Offline use would require staging the exact
checkpoint and all dependencies first, then using local paths. Its
[image loader](https://github.com/btsmart/splatt3r/blob/bda1dd07cf84b9189baabb1b8beac243959cfe5d/src/mast3r_src/dust3r/dust3r/utils/image.py)
with `our_method=True` resizes/crops to a square 512 grid. For a 512×640 image this
discards 128 horizontal pixels; a crop-only score is not our full-grid score.
Rectangular transformer patches may be possible, but complete 512×640 inference
and rendering are **not verified** by source inspection alone. MASt3R's metric
prior is relevant; pose-free input alone is not proof of arbitrary scale. The
actual checkpoint still needs a raw metric-frame check before using the known rig
without GT pose/scale fitting.

Licence evidence is separate for [Splatt3R](https://github.com/btsmart/splatt3r/blob/bda1dd07cf84b9189baabb1b8beac243959cfe5d/License),
[MASt3R](https://github.com/naver/mast3r/blob/f5209afc300cec36239a7ac992263f36847bbba0/LICENSE)
and [Graphdeco](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/LICENSE.md).
MASt3R is CC BY-NC-SA 4.0. Graphdeco uses its own research/noncommercial terms;
it must not be described as MIT because an integrating project is MIT. The
already prepared MVSplat/DepthSplat rasterizer fork has its own pinned notice at
`1250c420ebb945f0dce9945086e22faab9157c92`. These upstream identities are not a
claim that the latest standalone dependency HEAD equals the vendored version.

GPS's [pinned official README](https://github.com/aipixel/GPS-Gaussian/blob/0024776deee4824f270d4bb534a17ffd85f63cf2/README.md)
advertises a final Stage-2 checkpoint. We checked the official public listing,
not its human training dataset. No unverified third-party `gps_plus_final.pth`
mirror is substituted for original GPS-Gaussian. Native configs use resolution
1024 and human foreground masks; a full tissue/support mask is a disclosed domain
change. The source supports a rectified-pair bridge in principle, not a verified
512×640 offline run on this hardware.

## Fresh endoscopic reconstruction and depth search, 2025–2026

The search covered exact titles/IDs and combinations of “endoscopic/surgical”,
“feed-forward/generalizable”, “reconstruction/rendering/depth” and “Gaussian”.
Primary checks included arXiv pages/full text, MICCAI 2025 proceedings and 2026
EndoLINA pages, IPCAI/IJCARS, MIDL 2026, CVF proceedings/workshops, TMI and Displays.
This is a targeted survey, not a systematic-review claim. Search results were
leads; publication/protocol conclusions below use the linked primary sources.

| Additional candidate / primary source | Licence; official code + weights | Input → output and legitimate axis | Caveat; effort; recommendation |
| --- | --- | --- | --- |
| **Endo-PairGS**, IPCAI/IJCARS 2026, [DOI:10.1007/s11548-026-03707-y](https://doi.org/10.1007/s11548-026-03707-y), [official repo](https://github.com/MoriLabNU/Endo-PairGS) | Root CC BY-NC 4.0; MASt3R/rasterizer terms separately apply. Code and Drive-linked endoscopic Splatt3R fine-tune verified in README; weight bytes not downloaded. | Monocular **temporal frame pair** → aligned point maps/Gaussians; a subsequent Endo-4DGS stage fits a dynamic scene. The released `predict_2D.py` also renders the first-stage predictions. | **Important qualified candidate:** a stage-1-only R/Z experiment might be possible, labelled “Endo-PairGS initializer”, not full Endo-PairGS. Config already lists `[640,512]`; rig scale/frame, preprocessing and checkpoint completeness remain unverified. About 2–3 days for a bounded feasibility attempt. Full pipeline's per-scene rendering belongs in references. |
| **EndoLRMGS**, IEEE TMI 2026, [DOI:10.1109/TMI.2026.3707404](https://doi.org/10.1109/TMI.2026.3707404), [author manuscript record](https://discovery.ucl.ac.uk/id/eprint/10227896/), [official repo](https://github.com/MichaelWangGo/EndoLRMGS) | Root pipeline licence not verified; nested component licences do not license the whole project. Source released, OpenLRM general weights linked; a complete pretrained surgical pipeline checkpoint not verified. | Monocular video, depth priors and tool masks → optimized tissue Gaussians plus tool reconstruction; pose/scale optimization of tool priors. | **Reference only.** A feed-forward OpenLRM component does not make the complete scene pipeline feed-forward. R/Z under per-scene fitting cannot enter our zero-shot table. Over three days for a faithful port; do not pursue for deadline. |
| **Endo-FASt3r**, [arXiv:2503.07204](https://arxiv.org/abs/2503.07204), [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/4574_paper.pdf), [official repo](https://github.com/Mona-ShZeinoddin/Endo_FASt3r) | MIT root; incorporated foundation code separately licensed. Code and final/best official weights linked. No checkpoint downloaded. | Monocular pairs/video → depth and pose/reconstruction, no native RGB Gaussian renderer. **Z only**, conditional. | Default mono evaluation uses GT median scaling. Public source training list contains dataset identifiers 1–9, including 7; exact released-checkpoint lineage requires confirmation. Not a verified clean val_7 baseline. About 1–2 days technically; **reference/defer** scientifically. |
| **EndoDAV**, [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/1355_paper.pdf), [official repo](https://github.com/Zanue/EndoDAV) | MIT root. Code and model release advertised. `ckpts/models.zip` is a Git-LFS **pointer**, not a downloaded checkpoint archive; expected hash recorded below. | Monocular video → temporally consistent depth; **Z only**, no R. | Default evaluation has GT median scaling. Current source train lists include dataset 7; legacy/video lists also contain identifiers 8/9. They do not establish exact weight lineage. Native defaults differ from A00 resolution. About 1–2 days; **defer main raw-metric comparison**. |
| **SurgCUT3R**, [arXiv:2603.06971](https://arxiv.org/abs/2603.06971), [official repo](https://github.com/chumo-xu/SurgCUT3R) | Root CUT3R CC BY-NC-SA 4.0. Actual repo has code and two official Drive model links, despite author project page still saying code coming soon. | Monocular video → recurrent point maps, depth and cameras; **Z only**, not a native GS renderer. | Distinguish `demo.py` feed-forward prediction from `demo_ga.py` global alignment. Training-ID overlap with our split not verified. Extra temporal frames require authorization. About 2–3 days; **geometry reserve**, not immediate R baseline. |
| **EndoStreamDepth**, [arXiv:2512.18159](https://arxiv.org/abs/2512.18159), [MIDL 2026 paper](https://openreview.net/pdf?id=I7lgdDdcij), [official repo](https://github.com/MedICL-VU/EndoStreamDepth) | Apache-2.0 root; pretrained branch terms separate. Source and C3VD split-1/split-2 and SimCol3D weights linked. | Monocular frame stream with Mamba memory → claimed **metric depth**, no RGB renderer. **Z only**, with depth-unit audit. | More promising metric-depth fit than GT-aligned mono models. Choose the published model/dataset by a rule fixed before results, not by val_7. C3VD/SimCol training is documented; SCARED checkpoint lineage/overlap is not asserted. Native square grids, temporal context and Blackwell Mamba kernels need checking. About 2–3 days; **best new monocular-depth reserve**. |
| **PRISM**, IPCAI/IJCARS 2026, [DOI:10.1007/s11548-026-03669-1](https://doi.org/10.1007/s11548-026-03669-1), [article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13451345/), [official repo](https://github.com/XinweiJu/PRISM) | Root licence not verified. Code and v1.0.0 `PRISM-DLPE` weights archive plus SHA sidecar advertised. No archive downloaded. | RGB with luminance/shading priors for depth and RGB/edge inputs for pose → depth/pose, **Z only**. | Additional prior-generation dependencies; raw metric scale and split lineage need audit. No rendering metric. About 2–3 days conditional; **depth reserve**, behind released stereo predictors. |
| **ColonAdapter**, [arXiv:2511.22250](https://arxiv.org/abs/2511.22250), [official repo](https://github.com/JayJiang99/ColonAdapter) | Root licence not verified. Implementation present; README still lists pretrained-weight upload as TODO. | Monocular image pairs, DUSt3R-family adapter → point maps/depth; **Z conditional**, no native Gaussian RGB renderer verified. | Checkpoint and metric-scale protocol unresolved. Over three days if training is required; **defer**. |
| **Depth Any Endoscopy (DAE)**, [CVPR 2026 primary supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Shao_Depth_Any_Endoscopy_CVPR_2026_supplemental.pdf), [official repo](https://github.com/ShuweiShao/DAE) | MIT root, but inspected repository contains README/LICENSE and says source coming soon. Runnable implementation and weights **not verified**. | Monocular endoscopic depth estimation; **Z conditional**, no R. | A public GitHub URL alone is not a runnable release. More than three days to recreate; **cite, defer**. |
| **Endo3DAC**, [arXiv:2503.15917](https://arxiv.org/abs/2503.15917) | Official executable code, weights and licence not verified in sources checked; commit N/A. | Monocular depth, camera and pose predictions; reconstruction additionally uses depth scale/shift optimization. **Z conditional**, no native R verified. | Separate the network from the optimized reconstruction. No GT scale fit or per-scene optimization in our main table. More than three days without a release; **cite, defer**. |
| **SCOPE-4D**, [arXiv:2610.02343](https://arxiv.org/abs/2610.02343), [author project](https://chaoyizh.github.io/SCOPE-4D-page/) | Official code/weights/licence not verified on inspected primary pages; commit N/A. | Monocular video → feed-forward cameras, dense geometry and tissue tracks. **Z conditional**, not a verified Gaussian renderer. | Relevant generalizable reconstruction, different temporal input. More than three days without release; **cite, monitor**. |
| **ColScale3R**, [MICCAI 2026 EndoLINA paper page](https://papers.miccai.org/miccai-2026-sat/EndoLINA_015.html) | Official code, weights and licence not verified; commit N/A. | Adapted MapAnything using calibrated **three-view** colonoscopy → metric depth/point maps. **Z only**. | Requires more than our allowed pair and a different imaging context. More than three days absent release; **cite, not deadline run**. |
| **Adapting Generalizable Visual Geometry Models for Endoscopic 3D Reconstruction**, [MICCAI 2026 EndoLINA page](https://papers.miccai.org/miccai-2026-sat/EndoLINA_012.html) | Code, weights and licence not verified on primary page; commit N/A. | Adapted VGGT, depth/pose/reconstruction; **Z conditional**, no native R verified. | Distinct from withdrawn EndoVGGT. Release/split/scale audit needed; over three days without code; **cite**. |
| **Normal-guided cross-attention monocular depth**, IPCAI/IJCARS 2026, [DOI:10.1007/s11548-026-03720-1](https://doi.org/10.1007/s11548-026-03720-1), [primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13553636/) | Official executable code, weights and code licence not verified; commit N/A. | Monocular RGB with normal guidance → depth; **Z conditional**, no R. | Relative/scale protocol needs verification before millimetre scores. More than three days without release; **cite only**. |
| **EndoMetric**, [MICCAI 2025 page](https://papers.miccai.org/miccai-2025/0294-Paper1790.html) | Official executable code, weights and licence not verified on checked paper page; commit N/A. | Monocular near-light geometry/illumination constraints → depth/pose/metric reconstruction. **Z conditional**, no verified Gaussian R. | Scale recovery relies on its imaging/illumination assumptions, not an interchangeable passive-stereo predictor. Over three days without release; **scale-related reference**. |
| **EndoMUST**, [arXiv:2506.16017](https://arxiv.org/abs/2506.16017), [official IROS 2025 repo](https://github.com/BaymaxShao/EndoMUST) | Root licence not verified in checked files. Evaluation code and official SCARED-trained weights linked. | Monocular RGB → depth, plus pose/intrinsics estimation; **Z conditional**, no R. | Official command is mono evaluation; raw metric scale and SCARED sequence overlap need verification. README requests at least 24 GB for its setup, not measured inference memory. About 1–2 days for depth interface, but **defer until scale/split audit**. |
| **EndoDDC**, [arXiv:2602.21893](https://arxiv.org/abs/2602.21893), [official repo](https://github.com/Yinheng-Lin/EndoDDC) | Code directories are present although no root README/LICENSE was found in the checked paths. Endoscopy-trained inference weights and licence not verified. | RGB **plus sparse depth** → diffusion depth completion; **Z only**. | Additional sparse-depth input is outside the RGB-pair protocol. Do not supply GT samples as input. Over three days to define/port a matched condition; **exclude from current comparison**. |
| **EndoOmni** — older foundation-depth context, [arXiv:2409.05442](https://arxiv.org/abs/2409.05442), [official repo](https://github.com/TianCuteQY/EndoOmni) | Root licence not verified in checked paths. Code and two official OneDrive weight variants linked. | Single RGB → relative depth; variants explicitly described as scale-ambiguous or scale-and-shift-ambiguous. No R. | Default README evaluation uses median scaling. **Not eligible for raw metric MAE/EPE** without independently justified scale. About 1 day to interface, but scientific mismatch remains; cite as prior, not a 2025–26 new stereo baseline. |

The Endo-PairGS distinction matters for the opening search claim. Its
[released prediction config](https://github.com/MoriLabNU/Endo-PairGS/blob/9229618da01e2905050ed149a68e86aa176d7c77/configs/predict.yaml)
uses a Splatt3R checkpoint and rectangular resolution; its
[first-stage renderer](https://github.com/MoriLabNU/Endo-PairGS/blob/9229618da01e2905050ed149a68e86aa176d7c77/predict_2D.py)
exists. Thus a surgical feed-forward **component** with released weights is a
real lead. The complete paper method initializes and optimizes Endo-4DGS; it is
not another verified single-stereo-pair feed-forward protocol. A future head-only
experiment must disclose that component scope, temporal-to-stereo domain change,
which checkpoint includes which heads, and use our frozen masks/metrics instead
of the upstream prediction-dependent masking. No such experiment was performed.

### Scene-fitting references screened separately

| Candidate / primary source | Code/weights/licence finding | Inputs → outputs; protocol; effort/recommendation |
| --- | --- | --- |
| **SurGSplat**, [arXiv:2506.05935v2](https://arxiv.org/abs/2506.05935v2), [author page](https://surgsplat.github.io/) | Runnable release, weights and code licence not verified in sources checked; commit N/A. | Monocular sequence → progressively optimized cameras/GS and rendered views. R/geometry only in a **scene-fitting reference table**. Do not reimplement; over three days without release. |
| **Free-DyGS**, [arXiv:2409.01003v3](https://arxiv.org/abs/2409.01003v3) | Current abstract still promises future code; official executable release, weights and licence not verified; commit N/A. | Learned Sparse Gaussian Regressor initialization followed by framewise camera/deformation optimization and scene expansion. The initializer does not make the whole method feed-forward. **Reference only**, no deadline port. |

EndoNeRF, EndoGaussian, EndoGS and G-SHARP retain the reference-only classification
in the original report. EndoLRMGS and the **full** Endo-PairGS pipeline join this
group. No published per-scene scores were copied into a feed-forward table.

The Displays paper and unchanged official Endo-E2E-GS repository were rechecked:
[PII S0141938226000168 / DOI:10.1016/j.displa.2026.103353](https://www.sciencedirect.com/science/article/pii/S0141938226000168).
A separate released Endo-E2E-GS follow-up satisfying the same protocol was **not
verified in the sources checked**. This does not erase Endo-PairGS's partial fit.
CVPR 2026 DAE and the MICCAI/MIDL/IPCAI hits above extend the geometry literature,
not the count of verified native stereo Gaussian renderers. Targeted CVPR/ICCV
workshop searches did not verify another runnable endoscopic stereo renderer.

## Stereo matching: geometry axis only

There are at least **two official endoscopic-weight advertisements** worth
following up; neither is a locally verified checkpoint download. A third candidate
has trained-model links whose exact SCARED file remains unverified. These older
methods are useful geometry controls even though the new-method search targets
2025–2026. FoundationStereo remains our teacher reference, not an independent new
baseline. None provides native PSNR/SSIM/LPIPS for our rendering axis.

| Candidate / primary source | Licence; code + endoscopic weights | Inputs → outputs / allowed metrics | Protocol caveat; effort; recommendation |
| --- | --- | --- | --- |
| **Bidirectional Semi-supervised Dual-branch CNN**, [arXiv:2210.08291](https://arxiv.org/abs/2210.08291), [official repo](https://github.com/HK-Shi/Bidirectional-SemiSupervised-Dual-branch-CNN) | GPLv3. Inference code; explicit official **SCARED** and **SERV-CT** Drive model links. Bytes/accessibility and local SHA not verified. | Rectified stereo RGB → disparity/confidence. **G only** after native sign, padding and A00 grid audit. | Audit checkpoint training IDs before claiming val_7 unseen; a SERV-CT-trained option could avoid SCARED overlap, subject to lineage confirmation. Old Python 3.7/PyTorch ≥1.7 recipe needs modern compatibility. About 1–2 days; **first extra geometry candidate**. |
| **MSDESIS**, [DOI:10.1109/TMI.2022.3181229](https://doi.org/10.1109/TMI.2022.3181229), [official repo](https://github.com/dimitrisPs/msdesis) | MIT. Inference code and official “all models” OneDrive archive linked; README describes SCARED disparity training and surgical domain adaptation. Specific archive bytes/checkpoint hashes not verified. | Rectified stereo RGB → disparity and surgical-tool mask. **G only**, not native R. | Multiple phases/capacities; choose a declared official variant before seeing val_7, verify training sequence overlap and saved-disparity scaling versus raw tensor units. About 1–2 days after retrieval; **second geometry candidate**. |
| **HybridStereoNet**, *Deep Laparoscopic Stereo Matching with Transformers*, [arXiv:2207.12152](https://arxiv.org/abs/2207.12152), [official repo](https://github.com/XuelianCheng/HybridStereoNet) | Root licence not verified. Code and official Drive snapshot folder linked. Exact SCARED-trained checkpoint not verified; bundled `leastereo_sf` checkpoint is **not evidence of SCARED fine-tuning**. | Stereo RGB → disparity. **G only**. | Old PyTorch 1.6/CUDA 10.2 recipe and Apex training dependency; Blackwell compatibility untested. About 2–3 days, cut if native changes exceed that. **Third, conditional fallback**, not a third verified endoscopic checkpoint. |
| **StereoMamba**, [arXiv:2504.17401](https://arxiv.org/abs/2504.17401), [official repo](https://github.com/MichaelWangGo/StereoMamba) | Root licence not verified. Code and **SceneFlow** pretrained weights released; SCARED fine-tuned weights not verified. | Rectified stereo RGB → disparity; **G only**. | Do not call SceneFlow weights endoscopic weights merely because the paper evaluates SCARED. Mamba/VMamba kernels and new fine-tuning are high-risk under the cap/deadline. **Not counted among the 2–3 endoscopic-weight options; defer**. |

Before future downloads, select exact files by published recipe/availability, log
training-domain overlap, and hash bytes on D:. Do not select whichever checkpoint
scores best on dataset 7. More training is not part of Task C.

## Exact source and checkpoint provenance

Survey evidence root:

`/mnt/d/3DGS_runs/reliableendo-gs/baselines-20261031/survey-20261031T144437Z/`

The manifests `repositories*.json` and `remote-repositories.json` record public
source identities; `sources/` contains retrieved metadata/README/licence evidence.
Some repositories were shallow-cloned for static inspection; others had only
pinned README/licence files fetched. An exact hash here establishes inspected
source identity, **not** that an executable environment or all submodules exist.

| Official repository | Exact inspected Git commit |
| --- | --- |
| `wrld/Endo3R` | `8145c8e5535d2111974561e7d1cc06f1d008fe5e` |
| `Haechan21/StereoNeRF` | `c7fe2fc10ca57ed7a86e113f29e0b699507b09da` |
| `btsmart/splatt3r` | `bda1dd07cf84b9189baabb1b8beac243959cfe5d` |
| `aipixel/GPS-Gaussian` | `0024776deee4824f270d4bb534a17ffd85f63cf2` |
| `MoriLabNU/Endo-PairGS` | `9229618da01e2905050ed149a68e86aa176d7c77` |
| `MichaelWangGo/EndoLRMGS` | `e525284defbc853a05dd085c5a3bed3f463c9d03` |
| `Mona-ShZeinoddin/Endo_FASt3r` | `eba3e1fee70836d1277e0a0264f4566f3cb9e0b7` |
| `Zanue/EndoDAV` | `8a9681b43d9b3b1600ba5f389e1ef69120af37cf` |
| `chumo-xu/SurgCUT3R` | `0010ae4778e0e4b8b699c81034da3b3555fdc0a7` |
| `MedICL-VU/EndoStreamDepth` | `5abe89d9c0e09f64fdc5276d21bb5a34aa815cc6` |
| `XinweiJu/PRISM` | `3e938afd4cf72d2219ed774996886f7eba2b5dee` |
| `JayJiang99/ColonAdapter` | `861e56b471c1947d74336cdf88cc237e40080315` |
| `ShuweiShao/DAE` | `e5ba8c5fb11c0ae7f5832436d197ab2d46353bff` |
| `BaymaxShao/EndoMUST` | `a513154617777565ebe660be56ca29ef966268b8` |
| `Yinheng-Lin/EndoDDC` | `70d96a415115d87e6cc95937f86ddf323fbbb272` |
| `TianCuteQY/EndoOmni` | `36c5e18a9a5cd7cd05f7fd3504202d26c4eca164` |
| `HK-Shi/Bidirectional-SemiSupervised-Dual-branch-CNN` | `e2d0510d4e92f7d59ce30977eb669a6a7e7c1b9b` |
| `dimitrisPs/msdesis` | `496c6011c9e582a7f0279b6f66da0d1137610dd8` |
| `XuelianCheng/HybridStereoNet` | `54f56a14cec49ffea624d406c5c5750f39e7c976` |
| `MichaelWangGo/StereoMamba` | `c9d29e2ea31bc2bbfc0d1fc761dffae512d9d41d` |
| `naver/mast3r` — standalone licence check | `f5209afc300cec36239a7ac992263f36847bbba0` |
| `graphdeco-inria/gaussian-splatting` — standalone licence check | `54c035f7834b564019656c3e3fcc3646292f727d` |
| `Intelligent-Imaging-Center/Endo-E2E-GS` | `186fa2b4a2159b28393492f6df1aa444b54391a8` |
| `donydchen/mvsplat` — existing preparation, not redone | `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01` |
| `cvg/depthsplat` — existing preparation, not redone | `2dad25a7b9ba7c5537aba08463ef88d8511a06be` |

Project HEAD observed during survey:
`2be5ceaedd657d5b0785a580fd2269c8e8c97864`; the workspace already contained other
uncommitted work. No commit or push was made. This report is not a model-run
provenance record: no model wall time, peak GPU memory or performance metric was
measured here.

### Weights: advertised hashes versus verified local bytes

| Artifact | Evidence and status |
| --- | --- |
| Splatt3R `epoch=19-step=1200.ckpt` | Official [HF metadata](https://huggingface.co/api/models/brandonsmart/splatt3r_v1.0?blobs=true), model revision `abfe839937ef63a7e6417ef8a6d311d857cc560f`, size **3,231,028,127 bytes**, advertised LFS SHA-256 `963a0e9a992150e1509fcc952a45b22d24263167edc814238abcf0dd956e0b18`. **Metadata only; local checkpoint hash not verified.** |
| GPS `GPS-GS_stage2_final.pth` | Official README advertises Baidu/OneDrive. Fresh OneDrive access returned 403; Baidu returned HTTP 200 at a share-initialization page, not checkpoint bytes or a verified file listing. **Not downloaded; checkpoint SHA-256 unavailable.** See `source-fetches.json` and `source-fetches-final.json`; no access bypass or third-party replacement. |
| Endo3R released model | Official Drive file ID `11hbBHEqBWes4oK2e8OeNi2RM-QtzhKE0`; public metadata page retrieved. **No weight bytes or local checkpoint SHA-256**. A successful HTML page fetch is not an offline model test. |
| Endo-PairGS fine-tuned Splatt3R | Official README links a Drive fine-tune. **Not downloaded; checkpoint SHA-256 unverified.** Head/base completeness remains to be checked before any initializer experiment. |
| EndoDAV `ckpts/models.zip` | Git-LFS pointer advertises object SHA-256 `d32e9c459e5c1942cc4596dee02655aec2dd17528b05f437307864ea8a759521`, size **464,347,092 bytes**. **Pointer only, not a local archive hash verification.** |
| HybridStereoNet bundled `pretrained_ckpt/leastereo_sf/checkpoint/best.pth` | A **22,077,205-byte** tracked file arrived with source checkout under D:. Locally computed SHA-256 `4eff7a1005136a0caa19c88cb709ca0fd1dba17dce3a61985d9ae5992aef689a`. Not loaded; this SceneFlow-named initialization is not a verified endoscopic fine-tune. |
| Other linked weights in candidate tables | Not downloaded for this survey; local checkpoint SHA-256 **not available**. No hash is invented from a filename or README claim. |

No new intended model-weight download was needed for this survey; the bundled
HybridStereoNet file is the source-checkout exception, explicitly hashed above.
`source-model-files.json` records this scan. Existing MVSplat/DepthSplat checkpoint
hashes and CPU tests remain in the earlier run report; they were not redone.

The raw arXiv API response in `sources/arxiv-named.txt` requested `max_results=20`
for 13 named IDs and returned 13/13 records; parsed metadata is beside it.
`public-training-list-audit.json` records counts and SHA-256 of upstream source
training lists, not our dataset manifests. Direct public HTTPS/source inspection
was the fallback because the GitHub connector was unavailable. The earlier
paper-lookup workflow was reused for citation checking; no new implementation
skill or execution workflow was invoked.

## Existing environments: inspect and reuse only if adequate

All four requested environments exist under `/home/bkai/miniconda3/envs/`.
They were **read only**. No activation, package import, model load, package
installation, cache migration or CUDA check was performed. Inventory comes from
`conda-meta`, package metadata and module-file presence, saved to
`environment/inventory.json` under the evidence root. File presence is weaker
than an import/ABI test; “complete and runnable” is **not verified** for any env.

| Existing env | Observed contents | Adequacy / reuse recommendation |
| --- | --- | --- |
| `endo3r` | Python **3.11.16**, Conda PyTorch **2.5.1**, torchvision **0.20.1**, `pytorch-cuda=12.1`, NumPy **2.0.1**; roma, viser, gdown, evo, kornia, OpenCV, scipy, einops and torch binary files present. | All modules in the inspected Endo3R minimal list were found by filesystem check. **Best existing candidate for a future Endo3R CPU import check**, not declared GPU-ready. CUDA-12.1-era PyTorch/Blackwell compatibility is untested. Posed/Splatt3R stack incomplete: no dacite, hydra, e3nn, Lightning or rasterizer found. |
| `dust3r` | Python **3.11.17**, Conda PyTorch **2.5.1**, torchvision **0.20.1**, `pytorch-cuda=12.1`, NumPy **2.4.6**; roma/OpenCV/scipy/einops present. | Missing viser, gdown, evo and kornia from the inspected Endo3R list. No Lightning/dacite/e3nn/rasterizer found. **Not complete for these methods as inspected**; no modification proposed in Task C. |
| `bench_st3r` | Python **3.11.16**, torch **2.7.1+cu128**, torchvision **0.22.1+cu128**, NumPy **1.26.4**, hydra-core **1.3.7**, OmegaConf **2.3.1**, viser, roma/evo/kornia; editable SAM2 path present. | Plausible modern point-map inference base; no Lightning, dacite, e3nn, xformers or Gaussian rasterizer found. **Not a complete Splatt3R/posed-GS runtime**. Prefer later read-only validation before considering a new D: environment. |
| `bench_streamsplat` | Python **3.9.25**, torch **2.7.1+cu128**, torchvision **0.22.1+cu128**, NumPy **1.26.3**, xformers **0.0.31.post1** and binary file, OmegaConf **2.3.0**. | Python is **below project minimum 3.10**; no Lightning/dacite/e3nn/GS rasterizer found. Do not reuse as the project benchmark runner. Existing name does not prove StreamSplat or Splatt3R works. |

The `cuda-version=12.9` metadata in the first two envs does not mean their PyTorch
binary was compiled for CUDA 12.9; `pytorch-cuda=12.1` is separately recorded.
No claim of Blackwell support follows from a toolkit package name. Existing envs
on the WSL filesystem were neither relocated nor enlarged; all new survey
artifacts were written to D:, apart from this requested Markdown deliverable.

## Housekeeping observations — reported, not changed

**Optional dependency test.** The user reports
`tests/baseline/test_posed_gaussians.py::test_native_loader_rejects_changed_upstream_before_import`
fails with `ModuleNotFoundError: dacite` in `endo_e2e_claude`, but passes in the D:
venv. Static inspection corroborates that dacite is absent in that env and that
`build_posed_predictor` imports dacite/OmegaConf before checking upstream Git
identity. A minimal follow-up is a **test-local** `pytest.importorskip("dacite")`
(and OmegaConf if needed) for this optional native-loader test, retaining its
changed-upstream assertion in the equipped D: env. Do not blanket-skip the tiny
dependency-free adapter tests. An alternative later cleanup could move the
identity check before optional imports, but that changes runtime code and is
outside this survey. **Neither fix was applied or tested here.**

**CPU double free.** The user reports that
`test_a00_model.py::test_metric_centers_use_principal_offset...` reproduces the
double free only with **blank** `CUDA_VISIBLE_DEVICES=''`, and passes when the
variable is **unset**. Record these as distinct environments. A future isolated CPU
recheck should use `env -u CUDA_VISIBLE_DEVICES ... pytest ...`; this is a proposed
command, **not executed in Task C**. The observation narrows a reproducer; it does
not prove the underlying allocator/CUDA cause or that the full suite passes.
There was no code workaround, environment edit or hidden test-result change.

## Near/far and execution status

For MVSplat/DepthSplat, **recommend option (b), citation-only for this survey**.
Keep their original shortlist/preparation intact, but make no new benchmark run
or numeric near/far choice here. Their RE10K bounds live in their pose scale, not
automatically in A00 metres. Option (a) is a possible future engineering task:
pre-register a rule using rig calibration and a declared effective-disparity
operating range, including principal-point offsets, without val_7 GT/quality.
The baseline alone does not determine a unique far bound; the operating-range
assumption must be stated. No such rule or values were implemented or approved
by this document.

| Required status | Task C evidence |
| --- | --- |
| **Implemented** | This new addendum; pinned public-source/metadata evidence on D:. No repository code, adapter, test or environment changes. Original reports retained. |
| **Tested** | Public-source identity/licence inspection, arXiv version-history checks, weight-link/metadata checks, filesystem-only env inventory, and SHA-256 of the bundled model file. These are survey checks, not model tests. |
| **Not tested** | Inference, training, strict checkpoint loads, offline execution, rectangular rendering, camera/scale adapters, CPU/GPU unit tests, memory/latency, metric quality, disparity-shift slopes and actual environment import/ABI completeness. Earlier report's tests remain earlier evidence only. |
| **Blocked** | GPS official checkpoint retrieval/hash; absent/placeholder releases; Endo3R val_7 training overlap; ambiguous raw monocular scale and temporal context; one-pair StereoNeRF and 3.2 GiB feasibility. No test is reported green by omission. |
| **Planned** | If execution is requested later: resolve bounds before posed-baseline runs; hash exact downloaded weights; audit checkpoint split lineage and raw camera units; perform tiny CPU checks and one capped-GPU process only after resource checks/authorization. Optional-dependency test change and unset-variable reproducer are proposals only. |

## Worth adding for the deadline (about 2 weeks)

Rank reflects protocol fit, released artifacts and remaining engineering work,
**not** val_7 performance. Keep at most **two additional rendering-baseline runs**
as the immediate execution target; the existing MVSplat/DepthSplat shortlist is
still the lowest-risk starting point once its bounds are resolved. The rest are
substitutes or a separately labelled geometry expansion, not a request to run all.

| Rank | Candidate / comparison group | Estimated additional effort | Deadline recommendation / stop condition |
| --- | --- | --- | --- |
| **1** | **MVSplat RE10K** — general feed-forward R/Z | About 1 day after bounds/resource decision; existing CPU preparation already done | Retain original first choice. Known cameras, official local checkpoint and existing bridge. Stop if full-grid native rendering cannot meet memory cap without model changes. **Citation/prepared only in Task C.** |
| **2** | **DepthSplat-small RE10K** — general feed-forward R/Z | About 1–2 days after same decision | Retain original second choice; choose no new size/weight by validation quality. Same scale/bounds and full-resolution memory blockers. **Citation/prepared only in Task C.** |
| **3** | **HK-Shi dual-branch CNN** — geometry-only G | About 1–2 days after checkpoint access/lineage check | Best expansion if a geometry axis is needed. Prefer a documented non-overlapping endoscopic checkpoint, with raw disparity conventions verified. Stop on unresolved training overlap or incompatible legacy code. |
| **4** | **MSDESIS** — geometry-only G | About 1–2 days after archive access | Second geometry option; fix architecture/phase choice from official recipe before results. Stop if checkpoint training split or disparity scale cannot be established. |
| **5** | **Endo-PairGS initializer only** — surgical component R/Z | About 2–3 days, conditional | Most relevant new surgical Gaussian lead, but **not full-method reproduction**. Stage-1 rendering, rectangular config and fine-tuned weights exist. Stop if rig gauge/scale, checkpoint completeness or memory requires retraining/GT fitting. Do not run Endo-4DGS or fetch EndoNeRF data. |
| **6** | **GPS-Gaussian** — stereo R/G | About 1–2 days after accessible official weight | Could replace a general renderer if official weight access and hash resolve promptly. Keep blocked while inaccessible; do not spend days replacing download services or retraining a surrogate. |
| **7** | **Splatt3R** — general pose-free R/Z | About 2–3 days | Reserve for a clearly disclosed full-grid/raw-rig experiment; stop on required GT alignment or >3-day rectangular/native port. No cropped scores in the main table. |
| **8** | **EndoStreamDepth** — monocular metric Z | About 2–3 days | Most interesting newly found monocular-depth reserve; do not add a renderer. Need fixed checkpoint, depth units, authorized context and compatible kernels. |
| **Exclude/defer** | Endo3R on val_7; withdrawn EndoVGGT; StereoNeRF one-pair port; unreleased StereoSplat/StereoGaussians; per-scene full pipelines | Scientific mismatch or >3 days without release | Cite with exact caveats. No imported paper metrics, no GT fitting and no reimplementation solely to fill a table. |
