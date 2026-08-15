"""Small, dependency-light episode video writer used by direct evaluations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_name(value: str) -> str:
    normalized = _SAFE_NAME_RE.sub("_", str(value).strip()).strip("._-")
    return normalized or "unknown"


def compose_side_by_side_frame(
    camera_frames: Iterable[tuple[str, np.ndarray]],
    overlay_lines: Iterable[str] = (),
) -> np.ndarray:
    """Return one BGR frame with RGB camera views placed side by side."""

    normalized: list[tuple[str, np.ndarray]] = []
    for name, frame in camera_frames:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"camera frame {name!r} must have shape (H,W,3), got {image.shape}."
            )
        if image.dtype != np.uint8:
            raise TypeError(
                f"camera frame {name!r} must be uint8 RGB, got {image.dtype}."
            )
        normalized.append((str(name), image))
    if not normalized:
        raise ValueError("at least one camera frame is required.")

    target_height = max(frame.shape[0] for _, frame in normalized)
    views: list[np.ndarray] = []
    for name, frame in normalized:
        if frame.shape[0] != target_height:
            width = max(1, round(frame.shape[1] * target_height / frame.shape[0]))
            frame = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr,
            name,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        views.append(bgr)

    canvas = np.concatenate(views, axis=1)
    for index, line in enumerate(overlay_lines):
        cv2.putText(
            canvas,
            str(line),
            (8, 46 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


class EpisodeVideoWriter:
    """Lazily write one MP4 and return an auditable completion descriptor."""

    def __init__(self, output_dir: Path, episode_index: int, fps: float) -> None:
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {fps!r}.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_index = int(episode_index)
        self.fps = float(fps)
        self.pending_path = self.output_dir / f"episode_{self.episode_index:03d}_pending.mp4"
        self._writer: cv2.VideoWriter | None = None
        self._frame_size: tuple[int, int] | None = None
        self.frame_count = 0
        self.error: str | None = None

    def write(
        self,
        camera_frames: Iterable[tuple[str, np.ndarray]],
        overlay_lines: Iterable[str] = (),
    ) -> None:
        if self.error is not None:
            return
        try:
            frame = compose_side_by_side_frame(camera_frames, overlay_lines)
            height, width = frame.shape[:2]
            frame_size = (width, height)
            if self._writer is None:
                self._frame_size = frame_size
                self._writer = cv2.VideoWriter(
                    str(self.pending_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    frame_size,
                )
                if not self._writer.isOpened():
                    raise RuntimeError(
                        f"failed to open MP4 writer at {self.pending_path}."
                    )
            elif frame_size != self._frame_size:
                raise ValueError(
                    f"video frame size changed from {self._frame_size} to {frame_size}."
                )
            self._writer.write(frame)
            self.frame_count += 1
        except Exception as exc:
            self.error = str(exc)

    def finalize(
        self,
        *,
        success: bool,
        end_reason: str,
        expected_frames: int,
    ) -> dict:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        outcome = "success" if success else f"failure_{_safe_name(end_reason)}"
        final_path = self.output_dir / (
            f"episode_{self.episode_index:03d}_{outcome}.mp4"
        )
        if self.error is None:
            if self.frame_count == 0 or not self.pending_path.is_file():
                self.error = "video contains no frames"
            elif self.frame_count != int(expected_frames):
                self.error = (
                    f"video frames={self.frame_count} do not match "
                    f"env_steps={expected_frames}"
                )
            elif final_path.exists():
                self.error = f"refusing to overwrite existing video {final_path}"
            else:
                self.pending_path.rename(final_path)

        completed = self.error is None and final_path.is_file()
        path = final_path if completed else self.pending_path
        return {
            "enabled": True,
            "status": "complete" if completed else "partial",
            "path": str(path.resolve()) if path.exists() else None,
            "frames": int(self.frame_count),
            "fps": self.fps,
            "error": self.error,
        }
