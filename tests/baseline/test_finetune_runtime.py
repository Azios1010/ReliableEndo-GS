"""Fine-tuning runtime and memory policy: order rules, exact exposure count, resume equivalence, checkpoint helpers."""

import pytest
import torch
from torch import nn

from reliable_endo_gs.baseline.finetune.memory import (
    amp_safe_checkpoint,
    checkpoint_module,
    fp32_island,
)
from reliable_endo_gs.baseline.finetune.runtime import (
    FinetuneConfig,
    order_digest,
    replay_order,
    run_finetune,
    validate_train_order,
)
from reliable_endo_gs.baseline.finetune_feasibility import both_view_mse
from reliable_endo_gs.contracts.deployment import DeploymentStereoBatch, RectifiedStereoCamera

H, W = 4, 6
IDS = [f"dataset_{d}/keyframe_1/frame_data{k:06d}" for d in (1, 2, 3, 6) for k in range(3)]


def _config(exposures: int, **kwargs) -> FinetuneConfig:
    base = dict(
        kind="mvsplat",
        exposures=exposures,
        seed=7,
        lr_main=1e-2,
        lr_pretrained=None,
        weight_decay=0.01,
        grad_clip=0.5,
        checkpoint_every=2,
        log_every=1,
    )
    base.update(kwargs)
    return FinetuneConfig(**base)


class Fake(nn.Module):
    """encoder.weight drives a constant image: enough to exercise the loop without any upstream code."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 1, bias=False)
        self.decoder = None


def _forward(model: Fake, batch: DeploymentStereoBatch) -> torch.Tensor:
    level = torch.sigmoid(model.encoder.weight.sum() + batch.left.mean())
    return level * torch.ones(1, 2, 3, H, W, device=batch.left.device)


def _loader(fail_at: int | None = None):
    calls = {"n": 0}

    def load(sample_id: str) -> DeploymentStereoBatch:
        calls["n"] += 1
        if fail_at is not None and calls["n"] == fail_at:
            raise OSError("simulated crash")
        seed = IDS.index(sample_id)
        generator = torch.Generator().manual_seed(seed)
        k = torch.eye(3)[None]
        transform = torch.eye(4)[None]
        transform[:, 0, 3] = -0.005
        camera = RectifiedStereoCamera(k, k.clone(), transform, True)
        return DeploymentStereoBatch(
            torch.rand(1, 3, H, W, generator=generator),
            torch.rand(1, 3, H, W, generator=generator),
            camera,
            torch.ones(1, 1, H, W, dtype=torch.bool),
        )

    return load


def _run(tmp_path, exposures, ids, **kwargs):
    torch.manual_seed(0)
    model = Fake()
    return model, run_finetune(
        model,
        _config(exposures),
        ids,
        kwargs.pop("load", _loader()),
        _forward,
        both_view_mse,
        tmp_path,
        device=torch.device("cpu"),
        provenance={"test": True},
        prefetch=1,
    )


def test_replay_order_and_validation_rules():
    ordered = replay_order(IDS, 3)
    assert ordered == IDS * 3 and len(ordered) == 36
    validate_train_order(ordered, _config(36))
    with pytest.raises(ValueError, match="must equal exposures"):
        validate_train_order(ordered, _config(35))
    with pytest.raises(ValueError, match="forbidden datasets"):
        validate_train_order(["dataset_7/keyframe_1/frame_data000000"] * 4, _config(4))
    with pytest.raises(ValueError, match="forbidden datasets"):
        validate_train_order(["dataset_8/keyframe_1/frame_data000000"] * 4, _config(4))
    assert order_digest(IDS) == order_digest(list(IDS)) != order_digest(IDS[::-1])


def test_exact_exposure_count_final_artifact_and_no_overwrite(tmp_path):
    model, summary = _run(tmp_path, 6, IDS[:6])
    assert summary["exposures"] == 6 and summary["optimizer_steps_skipped"] == 0
    assert (tmp_path / "final.pt").exists() and (tmp_path / "FINAL.json").exists()
    assert not (tmp_path / "resume.pt").exists()
    rows = [line for line in (tmp_path / "train_log.jsonl").read_text().splitlines() if line]
    assert len(rows) == 6  # log_every = 1
    with pytest.raises(FileExistsError):
        _run(tmp_path, 6, IDS[:6])


def test_resume_after_a_crash_equals_an_uninterrupted_run(tmp_path):
    ids = IDS[:6]
    reference, _ = _run(tmp_path / "straight", 6, ids)
    crashed = tmp_path / "crashed"
    with pytest.raises(OSError):
        _run(
            crashed, 6, ids, load=_loader(fail_at=5)
        )  # steps 0..3 done, resume.pt written at 2 and 4
    assert (crashed / "resume.pt").exists()
    resumed, summary = _run(crashed, 6, ids)
    assert summary["resumed_from"] in (2, 4)
    assert torch.equal(resumed.encoder.weight, reference.encoder.weight)


def test_resume_refuses_a_different_order(tmp_path):
    ids = IDS[:6]
    with pytest.raises(OSError):
        _run(tmp_path, 6, ids, load=_loader(fail_at=5))
    with pytest.raises(RuntimeError, match="differs|does not match"):
        _run(tmp_path, 6, IDS[::-1][:6])


def test_non_finite_loss_stops_the_run(tmp_path):
    model = Fake()

    def bad_forward(m, b):
        return _forward(m, b) * float("nan")

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        run_finetune(
            model,
            _config(4),
            IDS[:4],
            _loader(),
            bad_forward,
            both_view_mse,
            tmp_path,
            device=torch.device("cpu"),
            provenance={},
            prefetch=1,
        )
    assert not (tmp_path / "final.pt").exists()


def test_fp32_island_promotes_half_inputs_and_disables_autocast():
    seen = {}

    def probe(a, b):
        seen["dtype"] = a.dtype
        seen["autocast"] = torch.is_autocast_enabled("cpu")
        return a @ b

    a, b = torch.eye(3).half(), torch.eye(3).half()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = fp32_island(probe)(a, b)
    assert seen == {"dtype": torch.float32, "autocast": False} and out.dtype == torch.float32


def test_checkpointing_changes_memory_not_values():
    torch.manual_seed(1)
    plain = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
    wrapped = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
    wrapped.load_state_dict(plain.state_dict())
    for layer in wrapped:
        checkpoint_module(layer)
    x = torch.randn(3, 4, requires_grad=True)
    y_plain, y_wrapped = plain(x), wrapped(x)
    assert torch.allclose(y_plain, y_wrapped)
    y_plain.sum().backward()
    y_wrapped.sum().backward()
    for p, q in zip(plain.parameters(), wrapped.parameters(), strict=True):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)
    with torch.no_grad():
        assert torch.allclose(plain(x), wrapped(x))  # inference path bypasses checkpointing


def test_amp_safe_checkpoint_matches_the_plain_call():
    linear = nn.Linear(3, 3)
    x = torch.randn(2, 3, requires_grad=True)
    direct = linear(x)
    via = amp_safe_checkpoint(lambda t: linear(t), (x,), linear.parameters(), True)
    off = amp_safe_checkpoint(lambda t: linear(t), (x,), linear.parameters(), False)
    assert torch.allclose(direct, via) and torch.allclose(direct, off)
    via.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
