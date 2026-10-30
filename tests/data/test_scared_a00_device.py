"""Device preparation preserves A00 inputs and the separate offline GT boundary."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from reliable_endo_gs.data import scared_a00
from reliable_endo_gs.data.scared_a00 import (
    ScaredA00Adapter,
    ScaredA00SampleError,
    build_a01_manifest,
)
from tests.data.test_scared_a00_adapter import _fixture


@pytest.fixture(params=("cpu", "cuda:0"))
def input_device(request: pytest.FixtureRequest) -> Iterator[torch.device]:
    device = torch.device(request.param)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device preparation requires a GPU")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield device
    finally:
        torch.set_num_threads(previous_threads)


def test_rgb_normalization_all_uint8_values(tmp_path: Path, input_device: torch.device) -> None:
    pixels = np.tile(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), (1, 1, 3))
    path = tmp_path / "rgb.png"
    Image.fromarray(pixels).save(path)
    old = scared_a00._read_image(path, image_range="uint8_0_255")
    prepared = scared_a00._read_image(path, image_range="uint8_0_255", device=input_device)
    assert torch.equal(old, prepared.cpu())


def test_disparity_dtype_conversion_matches_cpu_on_requested_device(
    tmp_path: Path, input_device: torch.device
) -> None:
    source = np.linspace(-1, 4, 20, dtype=np.float64).reshape(4, 5)
    source[0, :2] = np.nan, np.inf
    path = tmp_path / "double-disparity.tiff"
    tifffile.imwrite(path, source)
    expected = scared_a00._read_disparity(path)
    prepared = scared_a00._read_disparity(path, device=input_device)
    assert prepared.device == input_device
    assert prepared.dtype == torch.float32
    torch.testing.assert_close(prepared.cpu(), expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("output_hw", (None, (2, 2), (4, 4)))
def test_device_inputs_and_gt_match_canonical_cpu(
    tmp_path: Path, input_device: torch.device, output_hw: tuple[int, int] | None
) -> None:
    root = _fixture(tmp_path)
    data = root / "dataset_1/keyframe_1/data"
    pixels = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    Image.fromarray(pixels).save(data / "left_finalpass/frame_data000000.png")
    truth = np.arange(20, dtype=np.float32).reshape(4, 5)
    truth[0, 0], truth[1, 1], truth[2, 2] = np.nan, np.inf, -1.0
    tifffile.imwrite(data / "disparity/frame_data000000.tiff", truth)
    manifest = build_a01_manifest(root, tmp_path / "manifest.json", require_all_folds=False)
    adapter = ScaredA00Adapter(root, manifest)
    kwargs = dict(crop_xy_right=(1, 0), crop_hw=(4, 4), output_hw=output_hw)
    sample_id = manifest.entries[0].sample_id
    old = adapter.load(sample_id, include_supervision=True, **kwargs)
    prepared = adapter.load(sample_id, include_supervision=False, device=input_device, **kwargs)
    separate = adapter.load_supervision(prepared)
    assert prepared.supervision is None
    assert old.supervision is not None
    assert prepared.calibration_metadata == old.calibration_metadata
    assert prepared.batch.left.device == input_device
    assert separate.disparity.device == input_device
    assert torch.equal(prepared.batch.left.cpu(), old.batch.left)
    assert torch.equal(prepared.batch.right.cpu(), old.batch.right)
    assert torch.equal(prepared.batch.camera.K_L.cpu(), old.batch.camera.K_L)
    assert torch.equal(prepared.batch.camera.K_R.cpu(), old.batch.camera.K_R)
    assert torch.equal(prepared.batch.camera.T_R_from_L.cpu(), old.batch.camera.T_R_from_L)
    assert torch.equal(prepared.batch.support.cpu(), old.batch.support)
    assert torch.equal(separate.disparity.cpu(), old.supervision.disparity)
    assert torch.equal(separate.valid_mask.cpu(), old.supervision.valid_mask)


@pytest.mark.parametrize("image_hw", ((4, 4), (64, 80), (1024, 1280)))
def test_twofold_rgb_rounding_matches_cpu(
    input_device: torch.device, image_hw: tuple[int, int]
) -> None:
    # Nonconstant channels exercise rounding in both axes, including the real
    # training shape. Only synthetic pixels are used; this is not a benchmark.
    pixels = np.random.default_rng(731).integers(0, 256, (3, *image_hw), dtype=np.uint8)
    reference = torch.from_numpy(pixels.astype(np.float32) / 255.0)
    output_hw = (image_hw[0] // 2, image_hw[1] // 2)
    prepared = scared_a00._resize_rgb(reference.to(input_device), output_hw)
    expected = scared_a00._resize_rgb(reference, output_hw)
    assert prepared.device == input_device
    assert torch.equal(prepared.cpu(), expected)


def test_arbitrary_resize_is_explicitly_rejected_on_cuda_only(
    input_device: torch.device,
) -> None:
    image = torch.ones((3, 4, 4), dtype=torch.float32, device=input_device)
    if input_device.type == "cuda":
        with pytest.raises(ScaredA00SampleError, match="exactly twofold"):
            scared_a00._resize_rgb(image, (3, 3))
    else:
        assert scared_a00._resize_rgb(image, (3, 3)).shape == (3, 3, 3)


def test_cuda_twofold_resize_rejects_unpinned_reference_threads(
    input_device: torch.device,
) -> None:
    if input_device.type != "cuda":
        pytest.skip("the CPU reference thread restriction is CUDA-specific")
    image = torch.ones((3, 4, 4), dtype=torch.float32, device=input_device)
    torch.set_num_threads(2)
    try:
        with pytest.raises(ScaredA00SampleError, match="OMP_NUM_THREADS=1"):
            scared_a00._resize_rgb(image, (2, 2))
    finally:
        torch.set_num_threads(1)


def test_cuda_image_preparation_has_no_cpu_interpolation(
    tmp_path: Path, input_device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    if input_device.type != "cuda":
        pytest.skip("the no-CPU-interpolation contract is CUDA-specific")
    root = _fixture(tmp_path)
    manifest = build_a01_manifest(root, tmp_path / "manifest.json", require_all_folds=False)
    adapter = ScaredA00Adapter(root, manifest)
    interpolate = scared_a00.functional.interpolate

    def require_cuda(image: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        assert image.device.type == "cuda", "CUDA preparation must not resize on CPU"
        return interpolate(image, *args, **kwargs)

    monkeypatch.setattr(scared_a00.functional, "interpolate", require_cuda)
    sample = adapter.load(
        manifest.entries[0].sample_id,
        crop_hw=(4, 4),
        output_hw=(2, 2),
        include_supervision=False,
        device=input_device,
    )
    truth = adapter.load_supervision(sample)
    assert sample.batch.left.is_cuda
    assert truth.disparity.is_cuda
    assert sample.supervision is None


def test_separate_truth_never_decodes_rgb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fixture(tmp_path)
    manifest = build_a01_manifest(root, tmp_path / "manifest.json", require_all_folds=False)
    adapter = ScaredA00Adapter(root, manifest)
    sample = adapter.load(manifest.entries[0].sample_id, include_supervision=False)

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("offline truth must not decode RGB again")

    monkeypatch.setattr(scared_a00, "_read_image", forbidden)
    assert adapter.load_supervision(sample).valid_mask.all()
    assert sample.supervision is None


@pytest.mark.parametrize(
    "key,value",
    (
        ("manifest_hash", "wrong"),
        ("protocol_identity", "wrong"),
        ("frame_id", "wrong"),
        ("output_hw", [2, 2]),
        ("input_hw", [40, 50]),
        ("crop_xy_right", [0, 1]),
        ("crop_hw", [50, 50]),
        ("runtime_resampling", {}),
    ),
)
def test_separate_truth_rejects_mismatched_binding(tmp_path: Path, key: str, value: object) -> None:
    root = _fixture(tmp_path)
    manifest = build_a01_manifest(root, tmp_path / "manifest.json", require_all_folds=False)
    adapter = ScaredA00Adapter(root, manifest)
    sample = adapter.load(manifest.entries[0].sample_id, include_supervision=False)
    changed = replace(sample, calibration_metadata={**sample.calibration_metadata, key: value})
    with pytest.raises((ScaredA00SampleError, ValueError)):
        adapter.load_supervision(changed)


def test_separate_truth_rejects_changed_rgb_source(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    manifest = build_a01_manifest(root, tmp_path / "manifest.json", require_all_folds=False)
    adapter = ScaredA00Adapter(root, manifest)
    sample = adapter.load(manifest.entries[0].sample_id, include_supervision=False)
    path = root / "dataset_1/keyframe_1/data/left_finalpass/frame_data000000.png"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ScaredA00SampleError, match="source file changed"):
        adapter.load_supervision(sample)
