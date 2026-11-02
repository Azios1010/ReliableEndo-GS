# Extra baseline feasibility — 2026-10-31

Task A decision, written before adapter implementation or baseline inference (availability updated after CPU setup): shortlist
**MVSplat RE10K** and **DepthSplat-small RE10K**, zero-shot, in that order. These are
conditional engineering candidates, not verified runnable measurements. Native CUDA
rasterizer compatibility, the 3.2 GiB process budget, and explicit near/far bounds in
metres must pass before either is called runnable. Do not train a new model for this
deadline unless this decision is revisited explicitly.

Scope: targeted public-paper and official-code search, 2024–2026 publications (including
2023 preprints of CVPR 2024 papers), accessed 2026-10-31. Negative search findings mean
“not verified in the sources checked,” not proof that no release exists. Training
costs below are published recipes, not measurements on this machine. No published
quality scores are imported into our benchmark table.

## Candidate table

| Candidate | Verified source | License | Code + weights? | Needs → outputs | Feasible in ≤3 days? | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| GPS-Gaussian (CVPR 2024) | [arXiv:2312.02155](https://arxiv.org/abs/2312.02155); [official code](https://github.com/aipixel/GPS-Gaussian) | MIT code; Graphdeco rasterizer has separate terms | Code cloned; README links `GPS-GS_stage2_final.pth` via Baidu/OneDrive; file availability/hash not yet verified | Two rectified RGB views, foreground masks, calibration and target camera → flow/depth, pixel Gaussians, RGB renders | Potentially, **conditional** on weights and camera/flow bridge; SCARED training recipe is not provided | Close stereo comparison, but behind the shortlist because weight access and human foreground preprocessing add uncertainty. Do not label a fresh SCARED port a native reproduction. |
| GPS-Gaussian+ | [arXiv:2411.11363](https://arxiv.org/abs/2411.11363); [author project](https://yaourtb.github.io/GPS-Gaussian+); [release question](https://github.com/aipixel/GPS-Gaussian/issues/83) | Separate + implementation/weight license unverified; original MIT does not establish it | Original repository exists; a distinct + implementation and checkpoint were not verified | Calibrated sparse views → depth and Gaussian parameter maps, target RGB; optional depth supervision | No verified release to port; implementing paper is outside 3-day limit | Cite as related work; do not rename original GPS-Gaussian as “+”. |
| Stereo-GS (ACM MM 2025) | [arXiv:2507.14921](https://arxiv.org/abs/2507.14921); [author project](https://kevinhuangxf.github.io/stereo-gs/); [official code](https://github.com/kevinhuangxf/stereo-gs) | No root LICENSE found at inspected commit; MASt3R/other dependencies have their own terms | Code cloned; official `download.py` points to `kevinhuangxf/stereo-gs/model.safetensors` on HF; download not tested | Pose-free, object-centric multi-view images (released examples 4/6 views) → point maps, Gaussians, rendered RGB; no documented calibrated A00 disparity interface | **No** faithful two-view 512×640 metric protocol established | Exclude from runs. Name does not imply a rectified stereo disparity predictor. Two views, metric frame/scale and aspect ratio would need additional validation, potentially retraining. |
| StereoGaussians | [arXiv:2609.38592](https://arxiv.org/abs/2609.38592); [full text](https://arxiv.org/html/2609.38592v1) | Public paper license is not a code/weight license; latter unverified | Paper verified; official runnable code and released weights not found in paper/search | Calibrated rectified pair, frozen stereo features/depth → two-layer expanded-canvas Gaussians, novel RGB; stereo depth can potentially give disparity | **No** without release; reimplementation exceeds limit | Strong conceptual comparison, unavailable execution evidence. Do not infer availability from its experiments. |
| StereoSplat — Yugay et al. | [arXiv:2610.35902](https://arxiv.org/abs/2610.35902); [author project](https://vladimiryugay.github.io/stereosplat/) | Code/weights unverified | Project currently says “Coming Soon” and links paper only | Stereo pair and rig calibration, foundation stereo → disparity/depth, Gaussians, metric-scale novel RGB | **No** verified runnable release | Cite; do not reimplement in this deadline. Distinct from the other StereoSplat entries below. |
| StereoSplat / StereoSplat+ — Liu & Okutomi | [arXiv:2607.08808](https://arxiv.org/abs/2607.08808); [full text](https://arxiv.org/html/2607.08808v1) | Code/weight license unverified | No official runnable release verified in paper/search | Variable posed stereo pairs; + starts from one pair, generates/refines additional views → Gaussians, RGB/depth | **No** verified code/weights; diffusion integration is beyond scope | Related work only; do not confuse its base StereoSplat with Yugay et al. |
| MVSplat (ECCV 2024) | [arXiv:2403.14627](https://arxiv.org/abs/2403.14627), DOI `10.1007/978-3-031-72664-4_21`; [official code](https://github.com/donydchen/mvsplat) | MIT code; separate modified Graphdeco rasterizer terms | Code cloned; official Drive `re10k.ckpt` downloaded and strictly loaded on CPU; ACID not downloaded | Two RGB views, normalized K, OpenCV camera-to-world poses, near/far → depth, covariance Gaussians/SH, target RGB | **Conditional yes**, estimated 1–2 days; actual 512×640 memory unmeasured | **Rank 1**. Small, established general feed-forward comparator with explicit posed camera input. |
| DepthSplat (CVPR 2025) | [arXiv:2410.13862](https://arxiv.org/abs/2410.13862); [official code](https://github.com/cvg/depthsplat); [model zoo](https://github.com/cvg/depthsplat/blob/main/MODEL_ZOO.md) | MIT code; pretrained branches/rasterizer retain separate terms | Code cloned; official HF small two-view RE10K GS checkpoint downloaded and strictly loaded on CPU (model zoo labels it 37M) | Two or more RGB views, normalized K, OpenCV camera-to-world poses, near/far → metric-scale depth, Gaussians, rendered RGB | **Conditional yes**, estimated 1–2 days; CUDA/xformers and memory need checking | **Rank 2**, small GS checkpoint, not the depth-only checkpoint. Adds a monocular prior; no SCARED tuning. |
| pixelSplat (CVPR 2024) | [arXiv:2312.12337](https://arxiv.org/abs/2312.12337); [official code](https://github.com/dcharatan/pixelsplat) | MIT code; modified rasterizer separate | Code cloned; README links camera-ready and original weights; file downloads not tested | Posed RGB pair, K, near/far → sampled depth/Gaussians and target RGB; no native rectified disparity head | Potentially ≤3 days, but high-resolution epipolar attention and sampling add memory/integration risk | Not shortlisted: MVSplat is a smaller cost-volume counterpart with simpler deterministic depth extraction. |
| AnySplat (SIGGRAPH Asia 2025 / TOG) | [arXiv:2505.23716](https://arxiv.org/abs/2505.23716); [official code](https://github.com/InternRobotics/AnySplat) (OpenRobotLab URL redirects here) | MIT repository; check all pretrained branch terms separately | Code cloned; README uses HF `lhjiang/anysplat` | Uncalibrated images → predicted cameras, depth, Gaussians and RGB; known metric rig is not its native input interface | **Not established** under 20% GPU cap and known-camera metric protocol | Defer. Must resolve predicted-world to rig alignment without GT fitting; do not silently substitute estimated cameras for requested input cameras. |
| Endo-E2E-GS (Displays 2026) | [DOI:10.1016/j.displa.2026.103353](https://doi.org/10.1016/j.displa.2026.103353); [official code](https://github.com/Intelligent-Imaging-Center/Endo-E2E-GS) | Existing pinned integration; no new license conclusion | Already available per task: matched final Stage 2 + released SCARED-C Stage 1/2 | Stereo RGB/calibration → disparity, Gaussians, RGB | Already present | **Do not rerun**. Pinned `186fa2b4`, unmodified; native parity with paper **not verified**. |
| Leveling3D (2026, includes surgical cross-domain evaluation) | [arXiv:2603.16211](https://arxiv.org/abs/2603.16211); [full text](https://arxiv.org/html/2603.16211v1) | Code/weight license unverified | Official runnable release not verified | AnySplat + geometry-conditioned Stable Diffusion 2 + second reconstruction pass → enhanced RGB/Gaussians | **No** verified release; complex port exceeds limit | Relevant feed-forward-plus-generation work, not an additional endoscopy-specific stereo backbone. |

All shortlisted methods can *represent* two calibrated cameras. This is not evidence
that their published checkpoints accept this resolution within the memory cap or
generalize to tissue. 512×640 is a disclosed resolution/domain change from their
256×256 RE10K training. Rectification is compatible with their general posed-view
input. Depth-derived disparity must include the principal-point difference and use
the existing canonical stereo conversion; no GT scale fitting is allowed.

## Published training requirements and execution risks

- **GPS-Gaussian:** checked configs prescribe 40,000 Stage-1 steps (batch 6) and
  100,000 Stage-2 steps (batch 2), source resolution 1024. Stage 1 uses depth/flow
  supervision; Stage 2 uses novel target views. README describes hours of preprocessing
  and about 50 GB for the human dataset. Wall-clock training cost not verified here.
  SCARED's two input cameras do not automatically reproduce its novel-target recipe.
- **GPS-Gaussian+:** paper permits depth + rendering or rendering-only supervision.
  Runnable training cost and a distinct released checkpoint remain unverified.
- **Stereo-GS:** paper describes two-stage Objaverse training, 8 V100 32 GB GPUs,
  and approximately 100–200 epochs depending on Gaussian resolution. This is an
  object reconstruction recipe, not a calibrated surgical stereo recipe.
- **StereoGaussians:** paper specifies frozen stereo encoders and 30,000 predictor
  steps, batch 32 at 256²; larger-resolution experiments use 2/8 GPUs. It relies
  on static-scene teacher renderings with target cameras. Recreating this training
  source is outside scope; no local training cost measured.
- **StereoSplat (Yugay):** paper uses synthetic XRStereo and a foundation stereo
  backbone. Local training cost unverified; no executable release located.
- **StereoSplat+ (Liu):** KITTI-360, cost-volume + 3D-volume branches and a diffusion
  enhancer. No runnable local recipe verified; no cost estimate claimed.
- **MVSplat:** checked RE10K config uses 300,001 steps, batch 14; README specifies
  an A100 80 GB for training. Photometric supervision, no GT depth required.
- **DepthSplat:** README specifies four GH200 96 GB GPUs by default (also documents
  4×4090 or 1×A100 alternatives at equal total sample count). GS uses rendering
  supervision; separate depth checkpoints are subsequently GT-depth fine-tuned.
  Small two-view GS is preferred to reduce memory pressure. No retraining proposed.
- **pixelSplat:** learned depth distribution and photometric training on RE10K/ACID;
  complete training GPU-hours unverified. Existing weights remove training need.
- **AnySplat:** pretrained VGGT-based model plus decoder training on scene datasets;
  README provides distributed training. Local training time and peak inference memory
  unverified. Avoid GT-dependent similarity alignment.
- **Leveling3D:** trains a diffusion adapter on top of AnySplat; local compute and
  release availability unverified. Medical volume models such as
  [MedGSSR](https://william2ai.github.io/medgssr/) take MRI/CT, not stereo endoscopy.

## Other surgical methods: separate protocol, references only

Searches also returned the following. Their reconstruction of a sequence or fitting
of scene parameters prevents inclusion as zero-shot feed-forward competitors.

| Method | Verified source / official code | Protocol and availability |
| --- | --- | --- |
| EndoNeRF | [official repository](https://github.com/med-air/EndoNeRF) | Per-scene neural field. **No data download authorized**; no folder inspected. STTR-Light depth is a stereo reference, not GT; physical stereo baseline remains undocumented per task. |
| EndoGaussian | [arXiv:2401.12561](https://arxiv.org/abs/2401.12561); [official code](https://github.com/CUHK-AIM-Group/EndoGaussian) | Per-scene initialization/deformation fitting; repository advertises MIT, dependencies separately licensed. Paper reports short per-scene training, which is not zero-shot inference. |
| EndoGS | [arXiv:2401.11535](https://arxiv.org/abs/2401.11535); [author-linked code](https://github.com/HKU-MedAI/EndoGS) | Deformable per-scene GS; checkpoint and license details not audited for execution because excluded by task. |
| G-SHARP | [arXiv:2512.02482](https://arxiv.org/abs/2512.02482) | Scene/deformation fitting with GSplat. Paper's Apache-2.0 statement concerns GSplat; a separately executable G-SHARP code/weight release was not verified. |
| Free-DyGS | [arXiv:2409.01003](https://arxiv.org/abs/2409.01003) | Pretrained Sparse Gaussian Regressor initializes a scene, then camera/deformation parameters are optimized frame by frame. A learned initializer does not make the full protocol zero-shot; code/license/weights unverified. |
| EndoSparse | [arXiv:2407.01029](https://arxiv.org/abs/2407.01029); [MICCAI paper](https://papers.miccai.org/miccai-2024/277-Paper0791.html) | Foundation priors guide sparse-view scene fitting; not an independently verified two-view feed-forward release. |
| Free3R-GS | [DOI:10.1016/j.patcog.2026.113980](https://doi.org/10.1016/j.patcog.2026.113980) | SfM-free initialization and reflection repair for deformable 4DGS; “SfM-free” does not establish zero-shot inference. Runnable release not audited. |

No additional endoscopy-specific, standalone two-view feed-forward GS release was
verified in this bounded search. This is not an exhaustive absence claim. Leveling3D
is relevant cross-domain work. ExtraGS appeared in author/search results, but its
arXiv endpoint could not be verified in this session; treat its identifier/release
as **unverified**, and do not use it to justify a baseline implementation.

## Ranked shortlist and frozen decisions

1. **MVSplat RE10K:** a standard general feed-forward model, with camera calibration
   supplied directly and explicit depth available from its visualization output.
   Use official final released weights zero-shot. Estimate 1–2 engineering days,
   conditional on downloading weights and native renderer compilation. No newly
   trained GPS-style surrogate is substituted for an official method.
2. **DepthSplat-small RE10K:** directly downloadable GS checkpoint
   `depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth`, two-view training, 37M
   parameters per model zoo. Uses the same posed bridge but adds a monocular branch.
   Estimate 1–2 days if native dependencies work. Do not choose model size or checkpoint
   based on val_7 quality. No depth-only weights or per-scene optimization.

**Unresolved before real-data inference:** the checked MVSplat RE10K experiment specifies near/far 1–100, and
DepthSplat RE10K specifies 0.5–100, in their pose coordinate units. A00 is metric. The task explicitly
forbids guessing ambiguous units, so a question requesting fixed near/far in metres
has been sent to the user. No bound is to be chosen from val_7 GT. Upstream normalized
image coordinates sample pixel centers at `(u+0.5)/W`; the bridge must explicitly map
A00 integer-center intrinsics to that convention and test it on synthetic tensors.

## Retrieval provenance

Artifacts, clones, future environments and caches live under
`/mnt/d/3DGS_runs/reliableendo-gs/baselines-20261031/`. Only requested source adapters,
tests, entry points and these documentation files belong in the repository.
The six shallow official clones are recorded in
`research/repositories-20261031T063215Z.json`, including full Git hashes. Their source
was inspected without running training code. The official StereoSplat page is saved
as `research/stereosplat-project-20261031T063309Z.txt`.

The arXiv API request used `id_list` for the named candidates; its raw Atom response
is `research/arxiv-candidates-20261031T063309Z.txt`. The first request inadvertently
used arXiv's default `max_results=10`: totalResults=12, returned=10. This partial
response is retained, not represented as complete; direct abstract/full-text pages
above provide the remaining verifications. Skill parser:
`.agents/skills/paper-lookup/scripts/arxiv_atom.py`. Web searches used exact candidate
titles/IDs and the combinations `feed-forward endoscopic Gaussian`,
`feed-forward surgical Gaussian splatting`, `stereo endoscopy generalizable splatting`.
The GitHub connector returned “Unknown tool”; direct HTTPS and shallow public clones
were the fallback. No private data were sent to web services.

Procedural lookup assistance: Kassis et al., *Scientific Agent Skills: A Library of
Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://arxiv.org/abs/2609.00065)
(2026; current abstract record checked). This is a tooling citation, not evidence of
baseline feasibility or scientific performance.

CPU setup follow-up: both shortlisted native encoders passed random 64×64 CPU disparity checks. Native rendering, 512×640, and the 20% GPU budget remain untested. The full arXiv metadata retry (`research/arxiv-candidates-complete-20261031T064728Z.xml`, `max_results=20`) returned 12/12 records; parsed JSON is beside it. See `baseline_run_report_2026-10-31.md` for exact availability, hashes, failures and blocked measurements.
