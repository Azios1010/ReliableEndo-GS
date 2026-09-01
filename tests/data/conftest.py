"""Synthetic, non-medical SCARED-C layout fixtures."""

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

ENDOSCOPE_CALIBRATION = """%YAML:1.0
---
R: !!opencv-matrix
   rows: 3
   cols: 3
   dt: f
   data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
T: !!opencv-matrix
   rows: 1
   cols: 3
   dt: f
   data: [ -4., 0., 0. ]
M1: !!opencv-matrix
   rows: 3
   cols: 3
   dt: f
   data: [ 100., 0., 2., 0., 101., 1., 0., 0., 1. ]
D1: !!opencv-matrix
   rows: 1
   cols: 5
   dt: f
   data: [ 0., 0., 0., 0., 0. ]
M2: !!opencv-matrix
   rows: 3
   cols: 3
   dt: f
   data: [ 110., 0., 2., 0., 111., 1., 0., 0., 1. ]
D2: !!opencv-matrix
   rows: 1
   cols: 5
   dt: f
   data: [ 0., 0., 0., 0., 0. ]
"""

COLMAP_INTRINSICS = """%YAML:1.0
---
K: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 99., 0., 2., 0., 98., 1., 0., 0., 1. ]
image_width: 4
image_height: 3
"""


def create_scared_c_keyframe(
    root: Path,
    *,
    dataset_name: str = "dataset_1",
    keyframe_name: str = "keyframe_1",
    missing: Iterable[str] = (),
    image_size: tuple[int, int] = (4, 3),
) -> Path:
    """Create one small synthetic static keyframe and return its directory."""

    width, height = image_size
    keyframe = root / dataset_name / keyframe_name
    keyframe.mkdir(parents=True, exist_ok=True)
    left = np.zeros((height, width, 4), dtype=np.uint8)
    right = np.zeros((height, width, 4), dtype=np.uint8)
    left[..., 0] = np.arange(width, dtype=np.uint8)
    left[..., 1] = 20
    left[..., 2] = 40
    left[..., 3] = 255
    right[..., 0] = 60
    right[..., 1] = np.arange(width, dtype=np.uint8)
    right[..., 2] = 80
    right[..., 3] = 255
    xyz = np.ones((height, width, 3), dtype=np.float32)
    xyz[..., 0] *= 1.0
    xyz[..., 1] *= 2.0
    xyz[..., 2] *= 3.0
    if width > 1 and height > 1:
        xyz[0, 0, 0] = np.nan
        xyz[0, 1, 1] = np.nan
    files = {
        "Left_Image.png": lambda: Image.fromarray(left).save(keyframe / "Left_Image.png"),
        "Right_Image.png": lambda: Image.fromarray(right).save(keyframe / "Right_Image.png"),
        "left_depth_map.tiff": lambda: tifffile.imwrite(keyframe / "left_depth_map.tiff", xyz),
        "right_depth_map.tiff": lambda: tifffile.imwrite(
            keyframe / "right_depth_map.tiff", np.flip(xyz, axis=1).copy()
        ),
        "endoscope_calibration.yaml": lambda: (keyframe / "endoscope_calibration.yaml").write_text(
            ENDOSCOPE_CALIBRATION, encoding="utf-8"
        ),
        "intrinsics_colmap.yaml": lambda: (keyframe / "intrinsics_colmap.yaml").write_text(
            COLMAP_INTRINSICS, encoding="utf-8"
        ),
        "frame_log.json": lambda: (keyframe / "frame_log.json").write_text(
            json.dumps(
                {
                    "key": "1_1",
                    "total_frames_on_disk": 2,
                    "included_count": 2,
                    "excluded_count": 0,
                    "included_frames": [1, 2],
                    "excluded_frames": [],
                }
            ),
            encoding="utf-8",
        ),
    }
    missing_names = set(missing)
    for name, writer in files.items():
        if name not in missing_names:
            writer()
    return keyframe


@pytest.fixture
def scared_c_root(tmp_path: Path) -> Path:
    create_scared_c_keyframe(tmp_path)
    return tmp_path


@pytest.fixture
def make_keyframe():
    return create_scared_c_keyframe
