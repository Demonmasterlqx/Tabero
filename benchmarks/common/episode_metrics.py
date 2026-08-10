from __future__ import annotations

"""Pure helpers for per-episode force and step evaluation metrics."""

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from benchmarks.common.metrics import (
    compute_contact_force_metrics_from_13d,
    compute_contact_force_metrics_from_lr_forces,
    compute_topk_mean,
)


FORCE_METRIC_KEYS = (
    "squeeze_avg_pred",
    "squeeze_avg_meas",
    "squeeze_max_pred",
    "squeeze_max_meas",
    "ap_avg_pred",
    "ap_avg_meas",
    "ap_max_pred",
    "ap_max_meas",
)


def resolve_episode_termination(
    *,
    info: Any,
    terminated: bool,
    truncated: bool,
) -> tuple[str, str | None]:
    """Resolve a stable evaluator end reason from IsaacLab termination logs.

    IsaacLab auto-resets a terminated environment before ``env.step`` returns,
    but it preserves the triggering manager term under
    ``info['log']['Episode_Termination/<term>']``.  Damage terms take precedence
    over generic termination labels so a broken object cannot be reported as an
    unexplained failure.
    """

    if truncated:
        return "truncated", None
    if not terminated:
        return "running", None

    log = info.get("log", {}) if isinstance(info, dict) else {}
    if not isinstance(log, dict):
        return "terminated", None

    active_terms: list[str] = []
    prefix = "Episode_Termination/"
    for key, value in log.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        try:
            if hasattr(value, "item"):
                value = value.item()
            active = float(value) > 0.0
        except (TypeError, ValueError):
            active = bool(value)
        if active:
            active_terms.append(key[len(prefix) :])

    damage_terms = sorted(term for term in active_terms if term.startswith("object_damage_"))
    if damage_terms:
        return "object_damage", damage_terms[0]
    if active_terms:
        return "terminated", sorted(active_terms)[0]
    return "terminated", None


def extract_object_damage_details(
    *, info: Any, terminal_term: str | None, env_index: int = 0
) -> dict[str, Any] | None:
    """Extract a JSON-safe damage trigger snapshot from IsaacLab extras."""

    prefix = "object_damage_"
    if not terminal_term or not terminal_term.startswith(prefix):
        return None
    object_name = terminal_term[len(prefix) :]
    damage_root = info.get("object_damage", {}) if isinstance(info, dict) else {}
    if not isinstance(damage_root, dict):
        return None
    raw = damage_root.get(object_name)
    if not isinstance(raw, dict):
        return None

    def scalar(value: Any) -> Any:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[env_index]
            if hasattr(value, "item"):
                value = value.item()
        except (IndexError, TypeError, ValueError):
            return None
        return value

    return {
        "object_name": object_name,
        "max_squeeze_force": float(raw["max_squeeze_force"]),
        "consecutive_frames": int(raw["consecutive_frames"]),
        "consecutive_count": int(scalar(raw.get("consecutive_count"))),
        "measured_squeeze_force": float(scalar(raw.get("measured_squeeze_force"))),
    }


def extract_friction_snapshot(*, info: Any, env_index: int = 0) -> dict[str, Any] | None:
    """Extract the currently active gripper/object friction pair from reset extras."""

    root = info.get("physics_friction", {}) if isinstance(info, dict) else {}
    if not isinstance(root, dict) or not root:
        return None

    def scalar(value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[env_index]
            if hasattr(value, "item"):
                value = value.item()
            result = float(value)
        except (IndexError, TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def pair(value: Any) -> dict[str, float] | None:
        if not isinstance(value, dict):
            return None
        static_friction = scalar(value.get("static_friction"))
        dynamic_friction = scalar(value.get("dynamic_friction"))
        if static_friction is None or dynamic_friction is None:
            return None
        return {
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
        }

    gripper = pair(root.get("gripper"))
    objects = {
        name: parsed
        for name, raw in sorted(root.items())
        if name != "gripper" and (parsed := pair(raw)) is not None
    }
    if gripper is None and not objects:
        return None
    return {"gripper": gripper, "objects": objects}


def summarize_friction_statistics(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize configured friction samples over all recorded episodes."""

    scopes: dict[str, dict[str, list[float]]] = {}
    for episode in episodes:
        friction = episode.get("friction")
        if not isinstance(friction, dict):
            continue
        gripper = friction.get("gripper")
        if isinstance(gripper, dict):
            scopes.setdefault("gripper", {"static_friction": [], "dynamic_friction": []})
            for key in ("static_friction", "dynamic_friction"):
                value = gripper.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    scopes["gripper"][key].append(float(value))
        objects = friction.get("objects", {})
        if not isinstance(objects, dict):
            continue
        for object_name, pair in objects.items():
            if not isinstance(object_name, str) or not isinstance(pair, dict):
                continue
            scope_name = f"object:{object_name}"
            scopes.setdefault(scope_name, {"static_friction": [], "dynamic_friction": []})
            for key in ("static_friction", "dynamic_friction"):
                value = pair.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    scopes[scope_name][key].append(float(value))

    result: dict[str, Any] = {}
    for scope_name, fields in sorted(scopes.items()):
        scope_summary: dict[str, Any] = {}
        for field_name, values in fields.items():
            scope_summary[field_name] = {
                "count": len(values),
                "mean": _mean_or_none(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        result[scope_name] = scope_summary
    return result


def _float_values(values: Iterable[float]) -> list[float]:
    finite_values: list[float] = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite_values.append(value)
    return finite_values


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _topk_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(compute_topk_mean(values, frac=0.05))


def _stack_lr(values: Sequence[Any]) -> np.ndarray | None:
    if len(values) == 0:
        return None
    vectors: list[np.ndarray] = []
    for value in values:
        try:
            vector = np.asarray(value, dtype=np.float32).reshape(3)
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(vector)):
            vectors.append(vector)
    if not vectors:
        return None
    return np.stack(vectors, axis=0)


def summarize_episode_force_metrics(
    *,
    force_mode: str,
    env_steps: int,
    actions_13d: Sequence[Any] = (),
    squeeze_pred_values: Iterable[float] = (),
    squeeze_meas_values: Iterable[float] = (),
    ap_pred_values: Iterable[float] = (),
    ap_meas_values: Iterable[float] = (),
    fL_meas_values: Sequence[Any] = (),
    fR_meas_values: Sequence[Any] = (),
) -> dict[str, Any]:
    """Summarize one episode while preserving the legacy eight force metrics.

    ``force_mode`` is one of ``full`` (hybrid/tactile), ``measured_only``
    (binary), or ``not_applicable`` (diffik/osc).
    """

    if force_mode not in {"full", "measured_only", "not_applicable"}:
        raise ValueError(f"unsupported force_mode: {force_mode}")
    if env_steps < 0:
        raise ValueError(f"env_steps must be non-negative, got {env_steps}")

    squeeze_pred = _float_values(squeeze_pred_values)
    squeeze_meas = _float_values(squeeze_meas_values)
    ap_pred = _float_values(ap_pred_values)
    ap_meas = _float_values(ap_meas_values)

    action_array: np.ndarray | None = None
    if len(actions_13d) > 0:
        valid_actions: list[np.ndarray] = []
        for action in actions_13d:
            try:
                action_array_item = np.asarray(action, dtype=np.float32).reshape(13)
            except (TypeError, ValueError):
                continue
            if np.all(np.isfinite(action_array_item)):
                valid_actions.append(action_array_item)
        if valid_actions:
            action_array = np.stack(valid_actions)

    pred_metrics = None
    if action_array is not None:
        pred_metrics = compute_contact_force_metrics_from_13d(action_array)

    fL_meas = _stack_lr(fL_meas_values)
    fR_meas = _stack_lr(fR_meas_values)
    meas_metrics = None
    if fL_meas is not None and fR_meas is not None and fL_meas.shape == fR_meas.shape:
        meas_metrics = compute_contact_force_metrics_from_lr_forces(fL_meas, fR_meas)

    force_metrics: dict[str, float | None] = {key: None for key in FORCE_METRIC_KEYS}
    if force_mode == "full":
        force_metrics.update(
            {
                "squeeze_avg_pred": _mean_or_none(squeeze_pred),
                "squeeze_avg_meas": _mean_or_none(squeeze_meas),
                "squeeze_max_pred": None if pred_metrics is None else float(pred_metrics.squeeze_max),
                "squeeze_max_meas": _topk_or_none(squeeze_meas),
                "ap_avg_pred": None if pred_metrics is None else float(pred_metrics.external_norm_mean),
                "ap_avg_meas": _mean_or_none(ap_meas),
                "ap_max_pred": None if pred_metrics is None else float(pred_metrics.external_norm_max),
                "ap_max_meas": _topk_or_none(ap_meas),
            }
        )
    elif force_mode == "measured_only":
        # Preserve the legacy binary-mode convention: prediction-style aggregate
        # slots mirror the measured contact metrics because no force is predicted.
        force_metrics.update(
            {
                "squeeze_avg_pred": _mean_or_none(squeeze_pred),
                "squeeze_avg_meas": _mean_or_none(squeeze_meas),
                "squeeze_max_pred": None if meas_metrics is None else float(meas_metrics.squeeze_max),
                "squeeze_max_meas": None if meas_metrics is None else float(meas_metrics.squeeze_max),
                "ap_avg_pred": None if meas_metrics is None else float(meas_metrics.external_norm_mean),
                "ap_avg_meas": None if meas_metrics is None else float(meas_metrics.external_norm_mean),
                "ap_max_pred": None if meas_metrics is None else float(meas_metrics.external_norm_max),
                "ap_max_meas": None if meas_metrics is None else float(meas_metrics.external_norm_max),
            }
        )

    predicted_action_steps = 0 if pred_metrics is None else int(pred_metrics.num_steps)
    measured_force_steps = 0 if meas_metrics is None else int(meas_metrics.num_steps)
    force_samples = {
        "predicted_action_steps": predicted_action_steps,
        "measured_force_steps": measured_force_steps,
        "squeeze_pred_steps": len(squeeze_pred),
        "squeeze_meas_steps": len(squeeze_meas),
        "ap_pred_steps": len(ap_pred),
        "ap_meas_steps": len(ap_meas),
        "predicted_contact_steps": None if pred_metrics is None else int(pred_metrics.num_contact_steps),
        "measured_contact_steps": None if meas_metrics is None else int(meas_metrics.num_contact_steps),
        "predicted_contact_ratio": None if pred_metrics is None else float(pred_metrics.contact_ratio),
        "measured_contact_ratio": None if meas_metrics is None else float(meas_metrics.contact_ratio),
    }

    if force_mode == "not_applicable":
        force_status = "not_applicable"
        force_samples["coverage_ratio"] = None
    else:
        if force_mode == "full":
            required_counts = (
                predicted_action_steps,
                measured_force_steps,
                len(squeeze_pred),
                len(squeeze_meas),
                len(ap_pred),
                len(ap_meas),
            )
        else:
            required_counts = (measured_force_steps, len(squeeze_meas), len(ap_meas))
        minimum_count = min(required_counts, default=0)
        coverage_ratio = 1.0 if env_steps == 0 else min(1.0, float(minimum_count) / float(env_steps))
        force_samples["coverage_ratio"] = coverage_ratio
        force_status = "complete" if all(count == env_steps for count in required_counts) else "partial"

    return {
        "force_status": force_status,
        "force_samples": force_samples,
        **force_metrics,
    }


def aggregate_success_force_metrics(
    episodes: Sequence[dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Average each legacy metric independently over successful episodes."""

    aggregates: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for key in FORCE_METRIC_KEYS:
        values: list[float] = []
        for episode in episodes:
            if not episode.get("success"):
                continue
            value = episode.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            value = float(value)
            if math.isfinite(value):
                values.append(value)
        aggregates[key] = _mean_or_none(values)
        counts[key] = len(values)
    return aggregates, counts


def _summarize_step_group(episodes: Sequence[dict[str, Any]]) -> dict[str, int | float | None]:
    summary: dict[str, int | float | None] = {"episodes": len(episodes)}
    for field in ("env_steps", "inference_chunks"):
        values = [episode.get(field) for episode in episodes]
        valid = [int(value) for value in values if type(value) is int and value >= 0]
        summary[f"{field}_total"] = sum(valid)
        summary[f"{field}_mean"] = float(np.mean(valid)) if valid else None
        summary[f"{field}_min"] = min(valid) if valid else None
        summary[f"{field}_max"] = max(valid) if valid else None
    return summary


def summarize_step_statistics(episodes: Sequence[dict[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    """Return all/successful/failed step summaries."""

    return {
        "all": _summarize_step_group(episodes),
        "successful": _summarize_step_group([episode for episode in episodes if episode.get("success") is True]),
        "failed": _summarize_step_group([episode for episode in episodes if episode.get("success") is False]),
    }


def validate_episode_records(
    episodes: Sequence[dict[str, Any]],
    *,
    expected_count: int,
    replan_steps: int,
    max_inference_chunks: int | None = None,
) -> list[str]:
    """Return completeness warnings without changing evaluation success status."""

    warnings: list[str] = []
    indices = [episode.get("experiment_index") for episode in episodes]
    valid_indices = [index for index in indices if type(index) is int]
    if len(episodes) != expected_count:
        warnings.append(f"expected {expected_count} episode records, found {len(episodes)}")
    if len(valid_indices) != len(indices):
        warnings.append("one or more episode records have a non-integer experiment_index")
    if len(set(valid_indices)) != len(valid_indices):
        warnings.append("duplicate experiment_index values in episode records")
    if set(valid_indices) != set(range(expected_count)):
        warnings.append("episode records do not cover experiment_index 0..N-1 exactly")

    for episode in episodes:
        index = episode.get("experiment_index")
        env_steps = episode.get("env_steps")
        inference_chunks = episode.get("inference_chunks")
        label = f"episode {index}"
        if type(episode.get("success")) is not bool:
            warnings.append(f"{label} has invalid success={episode.get('success')!r}")
        if not isinstance(episode.get("end_reason"), str) or not episode.get("end_reason"):
            warnings.append(f"{label} has invalid end_reason={episode.get('end_reason')!r}")
        hdf5_episode_index = episode.get("hdf5_episode_index")
        if hdf5_episode_index is not None and type(hdf5_episode_index) is not int:
            warnings.append(
                f"{label} has invalid hdf5_episode_index={hdf5_episode_index!r}"
            )
        if type(env_steps) is not int or env_steps <= 0:
            warnings.append(f"{label} has invalid env_steps={env_steps!r}")
            continue
        if type(inference_chunks) is not int or inference_chunks <= 0:
            warnings.append(f"{label} has invalid inference_chunks={inference_chunks!r}")
            continue
        if max_inference_chunks is not None and inference_chunks > max_inference_chunks:
            warnings.append(
                f"{label} inference_chunks={inference_chunks} exceeds maximum {max_inference_chunks}"
            )
        minimum_steps = (inference_chunks - 1) * replan_steps + 1
        maximum_steps = inference_chunks * replan_steps
        if not minimum_steps <= env_steps <= maximum_steps:
            warnings.append(
                f"{label} env_steps={env_steps} is outside [{minimum_steps}, {maximum_steps}] "
                f"for inference_chunks={inference_chunks}"
            )
        force_status = episode.get("force_status")
        if force_status not in {"complete", "partial", "not_applicable"}:
            warnings.append(f"{label} has invalid force_status={force_status!r}")
        elif force_status == "partial":
            warnings.append(f"{label} has partial force coverage")
        force_samples = episode.get("force_samples")
        if not isinstance(force_samples, dict):
            warnings.append(f"{label} has invalid force_samples")
        for key in FORCE_METRIC_KEYS:
            value = episode.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                warnings.append(f"{label} has invalid {key}={value!r}")
    return warnings


__all__ = [
    "FORCE_METRIC_KEYS",
    "aggregate_success_force_metrics",
    "summarize_episode_force_metrics",
    "summarize_step_statistics",
    "validate_episode_records",
]
