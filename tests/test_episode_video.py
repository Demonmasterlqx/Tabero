import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.episode_video import (  # noqa: E402
    EpisodeVideoWriter,
    compose_side_by_side_frame,
)


def test_compose_side_by_side_frame_accepts_different_camera_heights():
    main = np.zeros((48, 64, 3), dtype=np.uint8)
    wrist = np.full((24, 32, 3), 127, dtype=np.uint8)

    frame = compose_side_by_side_frame(
        [("agentview", main), ("wrist", wrist)], ["target_single=13 N"]
    )

    assert frame.dtype == np.uint8
    assert frame.shape == (48, 128, 3)


def test_episode_video_writer_records_one_frame_per_env_step(tmp_path):
    writer = EpisodeVideoWriter(tmp_path, episode_index=0, fps=20.0)
    main = np.zeros((48, 64, 3), dtype=np.uint8)
    wrist = np.full((48, 64, 3), 127, dtype=np.uint8)

    for step in range(4):
        writer.write(
            [("agentview", main), ("wrist", wrist)],
            [f"step={step}", "latched=1"],
        )

    descriptor = writer.finalize(
        success=True,
        end_reason="success",
        expected_frames=4,
    )

    assert descriptor["status"] == "complete"
    assert descriptor["frames"] == 4
    assert descriptor["error"] is None
    video_path = Path(descriptor["path"])
    assert video_path.name == "episode_000_success.mp4"

    capture = cv2.VideoCapture(str(video_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        assert capture.get(cv2.CAP_PROP_FPS) == 20.0
    finally:
        capture.release()


def test_episode_video_writer_marks_frame_count_mismatch_partial(tmp_path):
    writer = EpisodeVideoWriter(tmp_path, episode_index=2, fps=20.0)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    writer.write([("agentview", frame)])

    descriptor = writer.finalize(
        success=False,
        end_reason="max_inference_steps",
        expected_frames=2,
    )

    assert descriptor["status"] == "partial"
    assert "do not match" in descriptor["error"]
