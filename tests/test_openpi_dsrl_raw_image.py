import ast
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "benchmarks/openpi/openpi_inference_client.py"


def _load_client_helpers(*names: str) -> dict:
    source = CLIENT_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    selected = [
        node for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    namespace = {"cv2": cv2, "np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CLIENT_PATH), "exec"), namespace)
    return namespace


def test_dsrl_raw_image_opt_in_adds_raw_256_image_and_bytes_alias():
    helpers = _load_client_helpers("_add_dsrl_raw_image", "_add_bytes_key_aliases")
    raw_image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    resized_image = np.zeros((224, 224, 3), dtype=np.uint8)
    payload = {"image": resized_image, b"image": resized_image}

    helpers["_add_dsrl_raw_image"](payload, raw_image, enabled=True)
    helpers["_add_bytes_key_aliases"](payload, ("dsrl_raw_image",))

    assert payload["image"].shape == (224, 224, 3)
    assert payload["dsrl_raw_image"] is raw_image
    assert payload[b"dsrl_raw_image"] is raw_image


def test_dsrl_raw_image_capture_selects_agentview_and_downsamples_official_camera_frame():
    capture = _load_client_helpers("_capture_dsrl_raw_image")["_capture_dsrl_raw_image"]
    wrist_frame = np.full((1, 512, 512, 3), 7, dtype=np.uint8)
    agentview_frame = np.full((1, 512, 512, 3), 13, dtype=np.uint8)

    assert capture("eye_in_hand_cam", wrist_frame, enabled=True) is None
    captured = capture("agentview_cam", agentview_frame, enabled=True)

    assert isinstance(captured, np.ndarray)
    assert captured.dtype == np.uint8
    assert captured.shape == (256, 256, 3)
    assert np.all(captured == 13)


def test_dsrl_raw_image_capture_is_inert_when_disabled():
    capture = _load_client_helpers("_capture_dsrl_raw_image")["_capture_dsrl_raw_image"]

    assert capture("agentview_cam", None, enabled=False) is None


def test_dsrl_raw_image_default_does_not_change_payload_or_validate_raw_image():
    add_dsrl_raw_image = _load_client_helpers("_add_dsrl_raw_image")["_add_dsrl_raw_image"]
    resized_image = np.zeros((224, 224, 3), dtype=np.uint8)
    payload = {"image": resized_image, b"image": resized_image}
    original_keys = set(payload)

    add_dsrl_raw_image(payload, None, enabled=False)

    assert set(payload) == original_keys
    assert "dsrl_raw_image" not in payload
    assert b"dsrl_raw_image" not in payload


@pytest.mark.parametrize(
    ("raw_image", "message"),
    [
        (None, "numpy.ndarray"),
        (np.zeros((256, 256, 3), dtype=np.float32), "dtype uint8"),
        (np.zeros((224, 224, 3), dtype=np.uint8), r"shape \(256, 256, 3\)"),
        (np.zeros((1, 256, 256, 3), dtype=np.uint8), r"shape \(256, 256, 3\)"),
    ],
)
def test_dsrl_raw_image_opt_in_strictly_validates_raw_image(raw_image, message):
    add_dsrl_raw_image = _load_client_helpers("_add_dsrl_raw_image")["_add_dsrl_raw_image"]

    with pytest.raises((TypeError, ValueError), match=message):
        add_dsrl_raw_image({}, raw_image, enabled=True)


def test_openpi_client_declares_dsrl_raw_image_opt_in_default_false():
    module = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "OpenpiClientArguments":
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and item.target.id == "send_dsrl_raw_image"
                ):
                    assert ast.literal_eval(item.value) is False
                    return

    pytest.fail("OpenpiClientArguments.send_dsrl_raw_image is missing")
