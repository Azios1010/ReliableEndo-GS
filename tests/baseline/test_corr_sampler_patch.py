"""Static validation and integrity tests for the external corr_sampler patch."""

import json
from pathlib import Path


def test_corr_sampler_patch_artifact_and_metadata_exist() -> None:
    patch_path = Path("patches/raft_stereo/0001-corr-sampler-scalar-type-compatibility.patch")
    readme_path = Path("patches/raft_stereo/README.md")
    meta_path = Path("patches/raft_stereo/patch_metadata.json")

    assert patch_path.is_file(), f"missing patch artifact at {patch_path}"
    assert readme_path.is_file(), f"missing patch documentation at {readme_path}"
    assert meta_path.is_file(), f"missing patch metadata at {meta_path}"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["classification"] == "BUILD_API_COMPATIBILITY_ONLY"
    assert meta["target_repository"] == "https://github.com/princeton-vl/RAFT-Stereo"
    assert meta["target_commit"] == "6068c1a26f84f8132de10f60b2bc0ce61568e085"
    assert meta["target_files"] == ["sampler/sampler_kernel.cu"]
    assert "forward(volume, coords, radius)" in meta["preserved_signatures"]
    assert "backward(volume, coords, grad_output, radius)" in meta["preserved_signatures"]
    assert meta["scientific_status"] == "unchanged_science"


def test_corr_sampler_patch_diff_is_strictly_minimal() -> None:
    patch_path = Path("patches/raft_stereo/0001-corr-sampler-scalar-type-compatibility.patch")
    patch_text = patch_path.read_text(encoding="utf-8")

    # Documented metadata in patch header
    assert "BUILD_API_COMPATIBILITY_ONLY" in patch_text
    assert "https://github.com/princeton-vl/RAFT-Stereo" in patch_text
    assert "6068c1a26f84f8132de10f60b2bc0ce61568e085" in patch_text

    # Diff checks: only touches sampler/sampler_kernel.cu
    assert "diff --git a/sampler/sampler_kernel.cu b/sampler/sampler_kernel.cu" in patch_text
    diff_lines = [line for line in patch_text.splitlines() if line.startswith("diff --git")]
    assert len(diff_lines) == 1

    # Extract lines strictly within diff hunks (starting with @@ and ending before footer)
    diff_body = patch_text.split("diff --git", 1)[1]
    if "\n-- \n" in diff_body:
        diff_body = diff_body.split("\n-- \n", 1)[0]
    elif "\n--\n" in diff_body:
        diff_body = diff_body.split("\n--\n", 1)[0]

    hunk_lines = [
        line
        for line in diff_body.splitlines()
        if not line.startswith("---")
        and not line.startswith("+++")
        and not line.startswith("diff --git")
        and not line.startswith("index")
        and not line.startswith("@@")
    ]
    removed_lines = [line[1:].strip() for line in hunk_lines if line.startswith("-")]
    added_lines = [line[1:].strip() for line in hunk_lines if line.startswith("+")]

    assert len(removed_lines) == 2
    assert len(added_lines) == 2

    assert all("volume.type()" in line for line in removed_lines)
    assert all("volume.scalar_type()" in line for line in added_lines)
    assert all("AT_DISPATCH_FLOATING_TYPES_AND_HALF" in line for line in removed_lines)
    assert all("AT_DISPATCH_FLOATING_TYPES_AND_HALF" in line for line in added_lines)


def test_corr_sampler_patch_applies_cleanly_to_reference_kernel() -> None:
    """Fail-closed static test verifying exact replacement on canonical kernel lines."""
    reference_forward_snippet = (
        '  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel", ([&] {\n'
        "    sampler_forward_kernel<scalar_t><<<blocks, threads>>>(\n"
    )
    reference_backward_snippet = (
        '  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_backward_kernel", ([&] {\n'
        "    sampler_backward_kernel<scalar_t><<<blocks, threads>>>(\n"
    )

    expected_forward_snippet = (
        '  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.scalar_type(), "sampler_forward_kernel", ([&] {\n'
        "    sampler_forward_kernel<scalar_t><<<blocks, threads>>>(\n"
    )
    expected_backward_snippet = (
        '  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.scalar_type(), "sampler_backward_kernel", ([&] {\n'
        "    sampler_backward_kernel<scalar_t><<<blocks, threads>>>(\n"
    )

    # Verify that replacing volume.type() with volume.scalar_type() transforms snippets as expected
    assert (
        reference_forward_snippet.replace("volume.type()", "volume.scalar_type()")
        == expected_forward_snippet
    )
    assert (
        reference_backward_snippet.replace("volume.type()", "volume.scalar_type()")
        == expected_backward_snippet
    )
