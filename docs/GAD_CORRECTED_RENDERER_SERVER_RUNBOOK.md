# GaD corrected-renderer reproduction — server runbook (commands only)

Executor: an agent on a rented single-GPU Linux server (x86_64, Ubuntu-like, root or passwordless sudo). The agent runs the commands below **in order**, checks each PASS line mechanically, and **never edits code, configs, snapshots or scripts**. Scientific design = `PROTOCOL_CR1.txt` (shipped in the bundle). Provenance of this file: written 2026-11-14 from the local session; scripts were CPU-tested locally (`py_compile`, `prepare_stage2.py` reproduces the historical recipe hashes byte-for-byte, `--mode validate --verify-inputs` passes); GPU steps are verified on the server by the gates below.

## 0. DECISIONS (the user edits these three lines BEFORE handing the file to the agent)

```bash
export STAGE1_TARGET_SAMPLES=131500   # 131500 = 10 epochs (primary recipe) | 65750 = 5-epoch prefix | 39450 = 3-epoch prefix (13150 samples per epoch)
export EPOCH3_READOUT=yes             # yes: also produce an EXPLORATORY Stage-2 readout from the epoch-3 checkpoint while Stage 1 is paused (section 7); no: skip
export PRIMARY_LINEAGE=server         # server | local   (the local Stage-1 run on the user's machine is a second lineage; the user must decide which one is primary and whether to stop the local one)
```

## 1. Rules for the agent (read first)

- Allowed (log each in `deviations.log`): installing OS/Python packages, creating directories/symlinks/users, `tmux`, retrying transient network/IO/OOM failures, creating files exactly as written in this runbook.
- STOP and report to the user (do not improvise) when: any gate prints FAIL; any file shipped in the bundle must be changed; a command needs a different path; hashes differ; a training run prints `"event": "failed"`; `DIVERGED` is printed. **Never restart a diverged run from a checkpoint, never re-seed, never "repair" a run.**
- Never touch test data (datasets 8/9) before section 8 and never run any `--split test` command before `check_validation_gates.py` exits 0.
- One GPU, one identical environment for every Stage-2 process (same `GAD_GPU_UUID`, data root and paths: the protocol fingerprint includes the expanded environment). No concurrent GPU work of yours while a Stage-1 or Stage-2 run is active. Exception: nothing else.
- Hard-coded absolute paths (`/mnt/d/...`, `/home/bkai/...`) are mirrored on the server on purpose; do not "fix" them.
- Timeline guide: setup + smoke + dry run 2–4 h; Stage 1 depends on GPU speed (historical 0.6 s/step => 10 epochs ~22 h, 3 epochs ~6.6 h); Stage 2 B/C ~1–2 h; validation ~0.5 h; test 2–4 h.

## 2. Session file (create once; `source` it in every new shell)

```bash
mkdir -p /tmp/cr1 && cat > /tmp/cr1/session.env <<'EOF'
export NS=/mnt/d/3DGS_runs/reliableendo-gs/corrected-renderer-20261113
export REP=/home/bkai/.local/share/reliableendo-gs/reports/corrected-renderer-20261113
export REPO=/mnt/d/workspace/huann/reliableendo-gs
export ROOT=/mnt/d/3DGS_datasets/preprocessing/SCARED_A00
export CACHE=/home/bkai/.local/share/reliableendo-gs/runs/kd-multiscale-full-20261020T163154Z/teacher-cache
export TEACHER=/home/bkai/.local/share/reliableendo-gs/teachers/foundation_stereo
export RASTER=/mnt/d/an_3dgs/Endo-E2E-GS/gaussian-splatting/submodules/diff-gaussian-rasterization
export PY=/home/bkai/miniconda3/envs/endo_e2e_gs/bin/python
export SRV=$REP/scripts/server
export B=$NS/server_bundle
export GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
dev() { mkdir -p $REP/logs 2>/dev/null; echo "$(date -Is) $*" | tee -a ${REP}/logs/deviations.log /tmp/cr1/deviations_pre.log >/dev/null; }
EOF
cat >> /tmp/cr1/session.env <<EOF
export STAGE1_TARGET_SAMPLES=${STAGE1_TARGET_SAMPLES}
export EPOCH3_READOUT=${EPOCH3_READOUT}
export PRIMARY_LINEAGE=${PRIMARY_LINEAGE}
EOF
source /tmp/cr1/session.env; echo "GPU_UUID=$GPU_UUID target=$STAGE1_TARGET_SAMPLES epoch3=$EPOCH3_READOUT primary=$PRIMARY_LINEAGE"
```

## 3. Server preflight and directory mirror (agent)

```bash
source /tmp/cr1/session.env
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version --format=csv,noheader | tee /tmp/cr1/gpu.txt
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1); [ "$GPU_MIB" -ge 12000 ] && echo "PASS gpu memory $GPU_MIB MiB" || echo "FAIL gpu memory < 12 GB (memory cap 0.35 needs ~4-6 GB)"
FREE_GB=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$FREE_GB" -ge 200 ] && echo "PASS disk free ${FREE_GB} GB" || echo "FAIL need >= 200 GB free on /"
nproc; free -g | head -2; lscpu | grep -E "Model name|MHz" | head -2
sudo -n true && echo "PASS sudo" || { [ "$(id -u)" = 0 ] && echo "PASS root" || echo "FAIL need root or passwordless sudo"; }
sudo apt-get update -y && sudo apt-get install -y build-essential ninja-build tmux rsync git curl bzip2 pigz
sudo mkdir -p /mnt/d/3DGS_datasets/preprocessing /mnt/d/3DGS_runs/reliableendo-gs /mnt/d/workspace/huann /mnt/d/an_3dgs /home/bkai /home/bkai/.cache/torch/hub /home/bkai/.local/share/reliableendo-gs
sudo chown -R "$(id -un)":"$(id -gn)" /mnt/d /home/bkai
mkdir -p $NS/server_bundle $ROOT $CACHE $TEACHER $(dirname $RASTER) /home/bkai/.local/share/reliableendo-gs/preparations
dev "S3 preflight done; gpu: $(cat /tmp/cr1/gpu.txt)"
```
PASS: the three PASS lines print (no FAIL). Then tell the user to run section 4 (local machine) and wait for "upload done".

## 4. LOCAL machine (the user, NOT the agent): upload

Set `SERVER`, `PORT` first. Large transfers are resumable (`--partial`); run them in `tmux`. Pause the local Stage-1 run first if the user wants it faster (decision `PRIMARY_LINEAGE`).

```bash
export SERVER=USER@HOST PORT=22
export NS=/mnt/d/3DGS_runs/reliableendo-gs/corrected-renderer-20261113 B=/mnt/d/3DGS_runs/reliableendo-gs/corrected-renderer-20261113/server_bundle ROOT=/mnt/d/3DGS_datasets/preprocessing/SCARED_A00
RS="rsync -a --partial --info=stats1,progress2 -e 'ssh -p $PORT'"
# L1 bundle (339 MB) + file lists + spot hashes + cache stats
eval $RS $B/ $SERVER:$NS/server_bundle/
# L2 dataset subset for Stage 1 / Stage 2 training and validation (46.7 GB images + small JSON)
eval $RS --files-from=$B/files_train.txt $ROOT/ $SERVER:$ROOT/
eval $RS --files-from=$B/files_val_sealed56.txt $ROOT/ $SERVER:$ROOT/
eval $RS --files-from=$B/files_reprojection_json.txt $ROOT/ $SERVER:$ROOT/
eval $RS --prune-empty-dirs --include='*/' --include='a00_*.json' --include='a00_provenance/***' --include='_provenance/***' --include='a01_manifest.json' --include='preprocessing_summary.json' --exclude='*' $ROOT/ $SERVER:$ROOT/
# L3 teacher cache (50 GB, copy whole: its manifest is hash-locked) and teacher directory (without its venv)
eval $RS /home/bkai/.local/share/reliableendo-gs/runs/kd-multiscale-full-20261020T163154Z/teacher-cache/ $SERVER:/home/bkai/.local/share/reliableendo-gs/runs/kd-multiscale-full-20261020T163154Z/teacher-cache/
eval $RS --exclude=venv /home/bkai/.local/share/reliableendo-gs/teachers/foundation_stereo/ $SERVER:/home/bkai/.local/share/reliableendo-gs/teachers/foundation_stereo/
# L4 test subset (52.8 GB) - may be uploaded later, while Stage 1 trains; it is NOT read before section 8
eval $RS --files-from=$B/files_test.txt $ROOT/ $SERVER:$ROOT/
echo "upload done"
```

## 5. Server: verify uploads, environment, native build, gates G1/G3 (agent)

```bash
source /tmp/cr1/session.env
# 5.1 verify the bundle hash, then install miniconda FIRST, then extract the bundle at / (mirrored paths)
cd $B && sha256sum -c SHA256SUMS --ignore-missing 2>&1 | grep -E "cr1_bundle|SPOT|cache_stats|files_" ; sha256sum -c SHA256SUMS --ignore-missing --quiet && echo "PASS bundle+lists hashes" || echo "FAIL hash mismatch"
cd /tmp && curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/bkai/miniconda3
sudo tar -C / -xzf $B/cr1_bundle.tar.gz && sudo chown -R "$(id -un)":"$(id -gn)" /mnt/d /home/bkai
test -f $REP/plan/PROTOCOL_CR1.txt && echo "PASS bundle extracted"
cat /tmp/cr1/deviations_pre.log >> $REP/logs/deviations.log 2>/dev/null || true
cd $REP/plan && sha256sum -c PROTOCOL_CR1.sha256 && sha256sum -c SCRIPTS_CR1_SERVER.sha256 && echo "PASS protocol and server scripts hashes"
# 5.2 science environment (same torch/numpy as the historical env; python minor must be 3.10)
/home/bkai/miniconda3/bin/conda create -y -n endo_e2e_gs -c conda-forge --override-channels python=3.10
$PY -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r $REP/env/requirements_endo_e2e_gs.txt
$PY -c "import torch,numpy,sys;print(sys.version.split()[0],torch.__version__,torch.version.cuda,torch.cuda.is_available(),numpy.__version__)" | tee /tmp/cr1/env.txt
grep -q "2.7.1+cu128 12.8 True 2.2.6" /tmp/cr1/env.txt && echo "PASS torch/cuda/numpy" || echo "FAIL expected 2.7.1+cu128 / 12.8 / True / 2.2.6"
dev "env: $(cat /tmp/cr1/env.txt)"
# 5.3 CUDA toolkit 12.8 for the native build (skip the conda install if nvcc 12.8 already exists)
nvcc --version 2>/dev/null | grep -q "release 12.8" || /home/bkai/miniconda3/bin/conda install -y -n endo_e2e_gs --override-channels -c nvidia/label/cuda-12.8.0 -c conda-forge cuda-nvcc cuda-cudart-dev cuda-libraries-dev
export CUDA_HOME=$( [ -x /usr/local/cuda/bin/nvcc ] && echo /usr/local/cuda || echo /home/bkai/miniconda3/envs/endo_e2e_gs ); export PATH=$CUDA_HOME/bin:$PATH
nvcc --version | grep "release 12.8" && echo "PASS nvcc 12.8" || echo "FAIL nvcc is not 12.8"
# 5.4 rebuild the pinned native rasterizer (source revision 59f5f77e + the 1-line <cstdint> include) for THIS GPU, register it with a .pth
cd $RASTER && rm -rf build *.egg-info diff_gaussian_rasterization/_C*.so
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
$PY setup.py build_ext --inplace 2>&1 | tee $REP/logs/native_build.log | tail -3
ls diff_gaussian_rasterization/_C*.so && echo "PASS native build"
echo $RASTER > $($PY -c "import site;print(site.getsitepackages()[0])")/diff_gaussian_rasterization_dev.pth
$PY -c "import diff_gaussian_rasterization as d,os;print(os.path.dirname(d.__file__))"
dev "native rasterizer rebuilt for sm_$TORCH_CUDA_ARCH_LIST; _C sha256 $(sha256sum diff_gaussian_rasterization/_C*.so | cut -c1-16) (historical build fb18b0abb7f91eef was sm_120)"
# 5.5 teacher venv at its mirrored path (thin venv over the science env, like the historical one)
$PY -m venv --system-site-packages $TEACHER/venv
grep -v '^-e ' $TEACHER/packages.txt > /tmp/cr1/teacher_packages.txt
$TEACHER/venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/cr1/teacher_packages.txt -c $TEACHER/constraints.txt
$TEACHER/venv/bin/python -c "import torch,omegaconf,timm,diff_gaussian_rasterization;print('PASS teacher venv',torch.__version__)"
```
PASS: every "PASS ..." line above prints. A FAIL stops the runbook.

```bash
source /tmp/cr1/session.env
# 5.6 data integrity (only the files the pipeline reads; test files are checked in section 8)
cd $ROOT
MISSING=0; for l in files_train files_val_sealed56 files_reprojection_json files_keyframe_json; do n=$(while read -r f; do [ -f "$f" ] || echo "$f"; done < $B/$l.txt | wc -l); echo "$l missing=$n"; MISSING=$((MISSING+n)); done
[ "$MISSING" = 0 ] && echo "PASS dataset lists complete" || echo "FAIL missing dataset files"
grep -v ' dataset_[89]/' $B/SPOT_SHA256.txt | sha256sum -c --quiet && echo "PASS train/val spot hashes (140 files)" || echo "FAIL spot hash"
EXP=$(grep manifest.json $B/cache_stats.txt | cut -c1-64); ACT=$(cd $CACHE && sha256sum manifest.json | cut -c1-64); [ "$EXP" = "$ACT" ] && echo "PASS cache manifest sha256" || echo "FAIL cache manifest sha256"
NT=$(cd $CACHE && find targets -type f | wc -l); [ "$NT" = 26995 ] && echo "PASS cache targets count 26995" || echo "FAIL cache targets count $NT (expected 26995)"
$PY -B - <<'EOF'
import hashlib, pathlib
for p, h in (("/mnt/d/3DGS_runs/reliableendo-gs/kd-multiscale-full-20261020T163154Z/training/initial.pt", "9d44b96f6e9597317a7b7386fbc36c449029a1c2cd737f150c83de54ea567a2d"),
             ("/home/bkai/.local/share/reliableendo-gs/runs/phase2-frozen-actions-20261028T114155Z/stage-a-attempt-00/final.pt", "ef7fa35f49e1ce21bd5994dedcb93b638750f98ca21afa68f37d1feb24fb4ba4"),
             ("/mnt/d/3DGS_runs/reliableendo-gs/clean10-20261103/kd/FINAL.pt", "db7413e8d456177ee55eb7e3cb75580b5cfd40c443403359f8173fb21a6131b6"),
             ("/home/bkai/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth", "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"),
             ("/home/bkai/miniconda3/envs/endo_e2e_claude/lib/python3.10/site-packages/lpips/weights/v0.1/alex.pth", "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0")):
    print("PASS" if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() == h else "FAIL", p.split("/")[-1])
EOF
```
PASS: no "missing" > 0, the 140-file spot check passes, cache manifest sha256 equals the expected value, all five checkpoints PASS.

```bash
source /tmp/cr1/session.env
# 5.7 GATE G1 on the server: real native backend, tolerance fixed in the test file (|mean| <= 0.01 px, every sample <= 0.05 px). pytest lives in a SEPARATE venv.
$PY -m venv --system-site-packages /home/bkai/pytest_venv && /home/bkai/pytest_venv/bin/pip install pytest
cd $REPO && CUDA_VISIBLE_DEVICES=$GPU_UUID PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$RASTER:$NS/stage2/source_snapshot/src /home/bkai/pytest_venv/bin/python -m pytest tests/rendering/test_native_pixel_convention.py -v -p no:cacheprovider -rs 2>&1 | tee $REP/logs/G1_regression_server.log | tail -12
grep -q " 8 passed" $REP/logs/G1_regression_server.log && ! grep -qi skipped $REP/logs/G1_regression_server.log && echo "PASS G1 (8 passed, none skipped)" || echo "FAIL G1 - STOP"
```

## 6. Amendment A2, Stage-1 smoke (G3), Stage-2 dry run, launch Stage 1

```bash
source /tmp/cr1/session.env
# 6.1 amendment A2 (hashed BEFORE any real training step)
cd $REP/plan && cat > AMENDMENT_CR1_A2_SERVER.txt <<EOF
AMENDMENT A2 to PROTOCOL_CR1 (written $(date -Is)): deployment on a rented server. Operational changes only; the protocol, snapshots, configs, seed, initial.pt, teacher cache, memory cap 0.35 and recipe are unchanged.
GPU: $(cat /tmp/cr1/gpu.txt). Python/torch/numpy: $(cat /tmp/cr1/env.txt). nvcc: $(nvcc --version | tail -1). Native rasterizer REBUILT for compute capability $TORCH_CUDA_ARCH_LIST from the pinned source (historical build sha prefix fb18b0abb7f91eef, sm_120): new _C sha256 $(sha256sum $RASTER/diff_gaussian_rasterization/_C*.so | cut -c1-64).
Environment rebuilt from requirements_endo_e2e_gs.txt (+ teacher packages.txt); paths /mnt/d and /home/bkai mirrored; the supervisor train_corrected_stage1_server.py differs from train_corrected_stage1.py in exactly one line (GPU UUID taken from KDATTR_GPU_UUID).
Stage-1 budget: STAGE1_TARGET_SAMPLES=$STAGE1_TARGET_SAMPLES (131500 = the 10-epoch recipe; any smaller value is a PREFIX of the same schedule, never a different recipe). Epoch-3 exploratory readout: $EPOCH3_READOUT (if yes: Stage 1 is paused at the epoch-3 checkpoint, Stage 2 + validation (and test only if stated by the user) run from it under its own TAG and run id, results are labelled EXPLORATORY; the full run remains the primary result).
Primary lineage: $PRIMARY_LINEAGE (a second Stage-1 run exists on the user's local machine). Data root: canonical SCARED_A00 for training and evaluation. External load on the server: none known / record in deviations.log.
EOF
sha256sum AMENDMENT_CR1_A2_SERVER.txt > AMENDMENT_CR1_A2_SERVER.sha256 && sha256sum -c AMENDMENT_CR1_A2_SERVER.sha256 && echo "PASS amendment A2 written and hashed"
# 6.2 GATE G3: Stage-1 smoke (40 samples, one resume) with the server supervisor
export KDATTR_GPU_UUID=$GPU_UUID KDATTR_SRC=$NS/stage1/source-snapshot/src
mv $NS/stage1/kd-smoke $NS/stage1/kd-smoke-from-local-bundle 2>/dev/null || true
cd $REPO && timeout 1800 python3 -B -u $SRV/train_corrected_stage1_server.py kd $NS/stage1 --smoke 40 --mem-fraction 0.35 2>&1 | tee $REP/logs/stage1_smoke_server.log | grep '"event"' | tail -4 | cut -c1-200
grep -q '"event": "complete"' $REP/logs/stage1_smoke_server.log && test -f $NS/stage1/kd-smoke/FINAL.pt && echo "PASS G3 smoke (FINAL.pt written, resume exercised)" || echo "FAIL G3 - STOP"
```
Note (verified in code, no action): Stage-1 training touches only the `final_train` manifest (`include_supervision=False`, `gt_weight` 0); it does not read validation data or GT disparity at epoch boundaries; epoch-end checkpoints are only saved.

```bash
source /tmp/cr1/session.env
# 6.3 Stage-2 DRY RUN on a scratch copy with the server smoke checkpoint (nothing here is kept)
export DRY=$NS/dry; rm -rf $DRY; mkdir -p $DRY && cp -a $NS/stage2/source_snapshot $DRY/source_snapshot
SMOKE=$NS/stage1/kd-smoke/FINAL.pt
$PY -B $SRV/prepare_stage2.py --final-ckpt $SMOKE --run-id cr1-dry --gpu-uuid $GPU_UUID --snapshot $DRY/source_snapshot --out $DRY/controls --run-root-base $DRY/runs 2>&1 | sed -n 1,5p | cut -c1-160
( set -a; . $DRY/controls/control_train/env.sh; set +a; cd $DRY
  PYTHONPATH=$DRY/source_snapshot/src $PY -B $DRY/source_snapshot/scripts/run_gad_decoupling.py $DRY/controls/control_train/configs --mode validate --verify-inputs | head -c 200; echo
  PYTHONPATH=$DRY/source_snapshot/src $PY -B $DRY/source_snapshot/scripts/run_gad_decoupling.py $DRY/controls/control_train/configs --mode initialize --allow-execution
  RUN=$DRY/runs/gad_decoupling_single_seed/cr1-dry; echo "shared_initial: $($PY -c "import json;print(json.load(open('$RUN/identity.json'))['shared_initial'])")  (historical 9b7d0f493b928f934b01879d2fea7ae791edab41df0b5199a2c39290a48ca032)"
  PYTHONPATH=$RUN/source_snapshot/src $PY -B $SRV/native_readiness.py $DRY/controls/control_train 2>&1 | tail -2 | cut -c1-300
  for arm in B C; do timeout 150 env PYTHONPATH=$RUN/source_snapshot/src $PY -B $DRY/source_snapshot/scripts/run_gad_decoupling.py $DRY/controls/control_train/configs --mode train --arm $arm --allow-execution > $DRY/train_$arm.log 2>&1; echo "arm $arm dry-run rows in train.jsonl: $(cat $RUN/*/attempt-*/train.jsonl 2>/dev/null | wc -l)"; done
  grep -h '"step"' $RUN/fixed_centres/attempt-*/train.jsonl | head -1 | cut -c1-120; grep -h '"step"' $RUN/trainable_centres/attempt-*/train.jsonl | head -1 | cut -c1-120 )
```
PASS: `validate` prints `"status": "configuration_valid"`; `initialize` ends without error; native readiness prints `"readiness": "PASSED"`; both arms have at least one `"step"` row (step 1 loss equal in B and C). Then clean up and launch:

```bash
source /tmp/cr1/session.env
rm -rf $NS/dry && echo "dry run removed"
# 6.4 launch Stage 1 (survives the session). Resume after any interruption = run the same tmux command again (idempotent).
tmux new -d -s cr1stage1 "bash $SRV/run_stage1_server.sh $STAGE1_TARGET_SAMPLES; sleep 86400"
sleep 60; tmux ls | grep cr1stage1 && echo "PASS stage1 launched"; tail -2 $REP/logs/stage1.log | cut -c1-200
# 6.5 monitor (agent: report to the user every few hours)
tail -1 $NS/stage1/kd/attempt-*/train.jsonl | tail -1 | $PY -c "import json,sys;r=json.loads(sys.stdin.read());c=r['cursor'];print('samples',c['samples_seen'],'of $STAGE1_TARGET_SAMPLES','rate',round(c['samples_seen']/r['elapsed_seconds'],2),'/s','ETA h',round(($STAGE1_TARGET_SAMPLES-c['samples_seen'])/(c['samples_seen']/r['elapsed_seconds'])/3600,1))"
grep -c transient $NS/stage1/kd/events.jsonl; python3 -c "import json;print(json.load(open('$NS/stage1/kd/status.json'))['event'])"
```
Stage 1 is finished when `$REP/done/CR1_STAGE1_DONE` and `$NS/stage1/kd/FINAL.pt` exist (`status.json` event `complete`). If `status.json` shows `failed`: STOP and report with the last 40 lines of `$REP/logs/stage1.log`.

## 7. OPTIONAL exploratory epoch-3 readout (only if `EPOCH3_READOUT=yes`)

When `$NS/stage1/kd/attempt-*/sample-000039450.pt` exists (epoch 3): stop Stage 1, run sections 9–12 with `TAG=ep3` and that checkpoint, then restart Stage 1 with the same command as 6.4 (it resumes from the newest checkpoint; at most ~1000 samples are lost).

```bash
source /tmp/cr1/session.env
until ls $NS/stage1/kd/attempt-*/sample-000039450.pt >/dev/null 2>&1; do sleep 300; done
tmux kill-session -t cr1stage1; sleep 5; pkill -f train_corrected_stage1_server.py; pkill -f capped_cli.py; sleep 10
pgrep -fa "capped_cli|train_corrected_stage1" && echo "FAIL processes still alive" || echo "PASS stage1 paused at epoch 3"
dev "Stage 1 paused at epoch-3 checkpoint for the exploratory readout"
export TAG=ep3 FINAL_CKPT=$(ls $NS/stage1/kd/attempt-*/sample-000039450.pt | head -1)
echo "now run sections 9, 10, 11 (validation only; test only if the user says so in writing) with TAG=$TAG; label results EXPLORATORY; then restart Stage 1 (6.4)"
```

## 8. After Stage 1: verification and Stage-1 per-epoch evaluation (GPU otherwise idle)

```bash
source /tmp/cr1/session.env
test -f $NS/stage1/kd/FINAL.pt && sha256sum $NS/stage1/kd/FINAL.pt | tee $REP/logs/stage1_FINAL.sha256 && echo "PASS FINAL.pt exists"
grep -c transient $NS/stage1/kd/events.jsonl | xargs echo "transient restarts:"
K=$((STAGE1_TARGET_SAMPLES/13150)); mkdir -p $NS/stage1_eval
for k in $(seq 1 $K); do n=$((k*13150)); ck=$(ls $NS/stage1/kd/attempt-*/sample-$(printf %09d $n).pt | head -1)
  CUDA_VISIBLE_DEVICES=$GPU_UUID OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$NS/stage2/source_snapshot/src $PY -B $SRV/eval_stage1_checkpoint.py $ck $NS/stage1_eval/ep$(printf %02d $k) 2>&1 | tail -1 | cut -c1-250
done
$PY -B $SRV/stage1_reading.py $NS/stage1_eval | tee $REP/logs/stage1_reading.txt
```
PASS: the table prints one row per epoch and the last line is `READING ...: STAGE1_REPRODUCES|STAGE1_BETTER|STAGE1_WORSE`. If it prints `DIVERGED` (exit code 2): STOP and report; do NOT retrain or repair.

## 9. Stage 2: prepare, initialize, readiness, train B then C (same GPU, sequential)

```bash
source /tmp/cr1/session.env
export TAG=${TAG:-full}; export FINAL_CKPT=${FINAL_CKPT:-$NS/stage1/kd/FINAL.pt}; export RUN_ID=cr1-s20261006-$TAG; export S2=$NS/stage2_$TAG
mkdir -p $S2 && cp -a $NS/stage2/source_snapshot $S2/source_snapshot
$PY -B $SRV/prepare_stage2.py --final-ckpt $FINAL_CKPT --run-id $RUN_ID --gpu-uuid $GPU_UUID --snapshot $S2/source_snapshot --out $S2/controls --run-root-base $S2/runs > $REP/logs/prepare_stage2_$TAG.log 2>&1; head -6 $REP/logs/prepare_stage2_$TAG.log | cut -c1-170
export RUN=$S2/runs/gad_decoupling_single_seed/$RUN_ID
( set -a; . $S2/controls/control_train/env.sh; set +a
  PYTHONPATH=$S2/source_snapshot/src $PY -B $S2/source_snapshot/scripts/run_gad_decoupling.py $S2/controls/control_train/configs --mode validate --verify-inputs | head -c 120; echo
  PYTHONPATH=$S2/source_snapshot/src $PY -B $S2/source_snapshot/scripts/run_gad_decoupling.py $S2/controls/control_train/configs --mode initialize --allow-execution
  echo "shared_initial: $($PY -c "import json;print(json.load(open('$RUN/identity.json'))['shared_initial'])")"
  PYTHONPATH=$RUN/source_snapshot/src $PY -B $SRV/native_readiness.py $S2/controls/control_train | tail -1 | cut -c1-300 )
```
PASS: `method check PASS`, `configuration_valid`, `initialize` without error, `"readiness": "PASSED"`. (The `shared_initial` equal to the historical `9b7d0f49…` is evidence of matched initialisation, not a gate.)

```bash
source /tmp/cr1/session.env; export TAG=${TAG:-full}; export S2=$NS/stage2_$TAG; export RUN=$S2/runs/gad_decoupling_single_seed/cr1-s20261006-$TAG
# B then C, strictly sequential, one tmux session (13,150 updates each; historical ~32 min each on a 5070 Ti)
tmux new -d -s cr1stage2 "( set -a; . $S2/controls/control_train/env.sh; set +a; for arm in B C; do PYTHONPATH=$RUN/source_snapshot/src $PY -B $S2/source_snapshot/scripts/run_gad_decoupling.py $S2/controls/control_train/configs --mode train --arm \$arm --allow-execution >> $REP/logs/stage2_train_\${arm}_$TAG.log 2>&1 || break; done; touch $REP/done/CR1_STAGE2_TRAIN_${TAG}_DONE ); sleep 3600"
until [ -f $REP/done/CR1_STAGE2_TRAIN_${TAG}_DONE ]; do sleep 120; done
ls $RUN/fixed_centres/final.pt $RUN/trainable_centres/final.pt && $PY -c "import json;[print(a,json.load(open('$RUN/'+d+'/summary.json'))['steps'],json.load(open('$RUN/'+d+'/summary.json'))['trainable_parameters']) for a,d in (('B','fixed_centres'),('C','trainable_centres'))]"
```
PASS: both `final.pt` exist; steps `13150`; parameters `1671` (B) and `1770` (C).

## 10. Validation (dataset_7, 56 sealed frames) and gates G5/G6

```bash
source /tmp/cr1/session.env; export TAG=${TAG:-full}; export S2=$NS/stage2_$TAG; export RUN=$S2/runs/gad_decoupling_single_seed/cr1-s20261006-$TAG
( set -a; . $S2/controls/control_eval/env.sh; set +a; export PYTHONPATH=$RUN/source_snapshot/src
  for arm in A B C; do $PY -B $S2/source_snapshot/scripts/run_gad_decoupling.py $S2/controls/control_eval/configs --mode evaluate --arm $arm --split validation --allow-execution || exit 1; done
  $PY -B $S2/source_snapshot/scripts/run_gad_decoupling.py $S2/controls/control_eval/configs --mode summarize --split validation --allow-execution )
$PY -B $SRV/check_validation_gates.py $RUN | tee $REP/logs/G56_validation_gates_$TAG.txt
```
PASS: every line prints `PASS` and the exit code is 0. **Any FAIL: STOP.**

Pre-registered validation diagnostics (eval-only; shift sweep for A, B, C, C-zero-offset under the corrected renderer; C projected displacement):

```bash
source /tmp/cr1/session.env; export TAG=${TAG:-full}; export S2=$NS/stage2_$TAG; export RUN=$S2/runs/gad_decoupling_single_seed/cr1-s20261006-$TAG
mkdir -p $S2/diag_validation
( set -a; . $S2/controls/control_eval/env.sh; set +a; export PYTHONPATH=$RUN/source_snapshot/src
  ATTR_SHIFTS="-0.5,-0.25,0.0,0.25,0.5,0.75,1.0" $PY -B $SRV/attribution_eval.py $S2/diag_validation $S2/controls/control_eval > $S2/diag_validation/run_stdout.txt 2>&1; tail -3 $S2/diag_validation/run_stdout.txt | cut -c1-160
  $PY -B $SRV/analyze_attribution.py $S2/diag_validation $RUN > $S2/diag_validation/analysis_stdout.txt 2>&1; tail -2 $S2/diag_validation/analysis_stdout.txt | cut -c1-160 )
$PY - <<EOF
import json;r=json.load(open("$S2/diag_validation/integrity.json"));print({k:v for k,v in r["reproduction_vs_existing_validation_rows_at_shift_0"].items()});print(r["integrity_all_frames"]);print("state unchanged:",r["state_digests_unchanged"])
EOF
```
PASS: reproduction block shows `bitwise_equal` = 1736 for A, B, C_full; `state unchanged: True`. (`B_attr_equals_C_attr` is expected to be `false`.)

## 11. Test (datasets 8/9) — ONLY after section 10 passed; frozen configuration; no selection

Skip this section for the exploratory `ep3` readout unless the user explicitly authorises test evaluation in writing.

```bash
source /tmp/cr1/session.env; export TAG=${TAG:-full}; export S2=$NS/stage2_$TAG; export RUN=$S2/runs/gad_decoupling_single_seed/cr1-s20261006-$TAG
test -s $REP/logs/G56_validation_gates_$TAG.txt && ! grep -q FAIL $REP/logs/G56_validation_gates_$TAG.txt && echo "PASS validation gates are green" || { echo "FAIL do not run test"; false; }
cd $ROOT && grep ' dataset_[89]/' $B/SPOT_SHA256.txt | sha256sum -c --quiet && echo "PASS test spot hashes (60 files)"
n=$(while read -r f; do [ -f "$f" ] || echo "$f"; done < $B/files_test.txt | wc -l); echo "test files missing: $n"; [ "$n" = 0 ] || echo "FAIL upload the test subset first"
( set -a; . $S2/controls/control_eval/env.sh; set +a; export PYTHONPATH=$RUN/source_snapshot/src
  # the first test evaluation writes test_freeze.json binding BOTH final checkpoints (and fails on any later change)
  for arm in A B C; do $PY -B $S2/source_snapshot/scripts/run_gad_decoupling.py $S2/controls/control_eval/configs --mode evaluate --arm $arm --split test --allow-execution || exit 1; done
  $PY -B $S2/source_snapshot/scripts/run_gad_decoupling.py $S2/controls/control_eval/configs --mode summarize --split test --allow-execution )
ls -l $RUN/test_freeze.json $RUN/evaluation/test/summary.json && echo "PASS test evaluated once"
```
Each arm takes about 80 min on a 5070 Ti-class GPU (5,907 frames); use `tmux` for the loop if the session may drop (rerunning an arm is refused if its output directory exists: STOP and report instead of deleting anything).

## 12. Runtime benchmark (GPU must be idle), results package

```bash
source /tmp/cr1/session.env; export TAG=${TAG:-full}; export S2=$NS/stage2_$TAG; export RUN=$S2/runs/gad_decoupling_single_seed/cr1-s20261006-$TAG
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
( set -a; . $S2/controls/control_eval/env.sh; set +a; export PYTHONPATH=$RUN/source_snapshot/src
  $PY -B $SRV/bench_runtime.py $S2/controls/control_eval $S2/runtime_$TAG.json )
cat $S2/runtime_$TAG.json | head -30
bash $SRV/package_results.sh /home/$(id -un)/cr1_results_$TAG
```
PASS: `runtime_*.json` has mean/median/p95 for A, B and C on the same eager FP32 path; `package_results.sh` prints a `SHA256SUMS` line and the archive size. Report the archive path and hash to the user.

## 13. What the agent hands back (results, not conclusions)

`cr1_results_<TAG>/cr1_results.tar.gz` + `SHA256SUMS`: Stage-1 log/events/epoch checkpoints, `stage1_eval/` and `stage1_reading.txt`, Stage-2 run roots (`identity.json`, `fixed_centres/`, `trainable_centres/`, `evaluation/{validation,test}/…/rows.jsonl`, `summary.json`, `test_freeze.json`), `diag_validation/` (tables, `analysis.json`, `curves_*.png`, `displacement_stats.json`), `runtime_*.json`, `G56_validation_gates_*.txt`, `G1_regression_server.log`, `deviations.log`, `PROTOCOL_CR1.txt`, `AMENDMENT_CR1_A1/A2`. Interpretation, historical-versus-corrected comparison, baseline audit and manuscript recommendations are written afterwards by the user's local assistant; the agent does not interpret, does not edit any paper file, and does not run additional experiments.
