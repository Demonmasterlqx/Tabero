import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.episode_metrics import (
    FORCE_METRIC_KEYS,
    aggregate_success_force_metrics,
    summarize_episode_force_metrics,
    summarize_step_statistics,
    validate_episode_records,
)


def _force_sequences(num_steps: int = 20):
    strengths = np.arange(1, num_steps + 1, dtype=np.float32)
    left = np.stack([strengths, np.zeros_like(strengths), strengths], axis=1)
    right = np.stack([np.zeros_like(strengths), np.zeros_like(strengths), -strengths], axis=1)
    actions = np.zeros((num_steps, 13), dtype=np.float32)
    actions[:, 7:10] = left
    actions[:, 10:13] = right
    squeeze = 2.0 * strengths
    applied = strengths
    return actions, left, right, squeeze, applied


def test_full_episode_force_summary_preserves_eight_legacy_formulas():
    actions, left, right, squeeze, applied = _force_sequences()

    summary = summarize_episode_force_metrics(
        force_mode="full",
        env_steps=20,
        actions_13d=actions,
        squeeze_pred_values=squeeze,
        squeeze_meas_values=squeeze,
        ap_pred_values=applied,
        ap_meas_values=applied,
        fL_meas_values=left,
        fR_meas_values=right,
    )

    assert summary["force_status"] == "complete"
    assert summary["squeeze_avg_pred"] == pytest.approx(21.0)
    assert summary["squeeze_avg_meas"] == pytest.approx(21.0)
    assert summary["squeeze_max_pred"] == pytest.approx(40.0)
    assert summary["squeeze_max_meas"] == pytest.approx(40.0)
    assert summary["ap_avg_pred"] == pytest.approx(10.5)
    assert summary["ap_avg_meas"] == pytest.approx(10.5)
    assert summary["ap_max_pred"] == pytest.approx(20.0)
    assert summary["ap_max_meas"] == pytest.approx(20.0)
    assert summary["force_samples"]["predicted_action_steps"] == 20
    assert summary["force_samples"]["measured_force_steps"] == 20
    assert summary["force_samples"]["predicted_contact_steps"] == 20
    assert summary["force_samples"]["measured_contact_steps"] == 20
    assert summary["force_samples"]["coverage_ratio"] == pytest.approx(1.0)


def test_top_five_percent_uses_nonzero_frames_and_all_zero_force_is_complete():
    actions = np.zeros((5, 13), dtype=np.float32)
    forces = np.zeros((5, 3), dtype=np.float32)
    zeros = np.zeros(5, dtype=np.float32)

    summary = summarize_episode_force_metrics(
        force_mode="full",
        env_steps=5,
        actions_13d=actions,
        squeeze_pred_values=zeros,
        squeeze_meas_values=zeros,
        ap_pred_values=zeros,
        ap_meas_values=zeros,
        fL_meas_values=forces,
        fR_meas_values=forces,
    )

    assert summary["force_status"] == "complete"
    assert all(summary[key] == pytest.approx(0.0) for key in FORCE_METRIC_KEYS)
    assert summary["force_samples"]["predicted_contact_steps"] == 0
    assert summary["force_samples"]["measured_contact_steps"] == 0
    assert summary["force_samples"]["predicted_contact_ratio"] == pytest.approx(0.0)


def test_missing_force_samples_are_partial_and_not_applicable_mode_keeps_steps():
    actions, left, right, squeeze, applied = _force_sequences(num_steps=3)
    partial = summarize_episode_force_metrics(
        force_mode="full",
        env_steps=3,
        actions_13d=actions[:2],
        squeeze_pred_values=squeeze[:2],
        squeeze_meas_values=squeeze,
        ap_pred_values=applied[:2],
        ap_meas_values=applied,
        fL_meas_values=left,
        fR_meas_values=right,
    )
    not_applicable = summarize_episode_force_metrics(
        force_mode="not_applicable",
        env_steps=3,
    )

    assert partial["force_status"] == "partial"
    assert partial["force_samples"]["coverage_ratio"] == pytest.approx(2 / 3)
    assert not_applicable["force_status"] == "not_applicable"
    assert not_applicable["force_samples"]["coverage_ratio"] is None
    assert all(not_applicable[key] is None for key in FORCE_METRIC_KEYS)


def test_success_only_force_aggregation_and_step_groups_use_independent_counts():
    successful = {
        "success": True,
        "env_steps": 11,
        "inference_chunks": 2,
        **{key: float(index + 1) for index, key in enumerate(FORCE_METRIC_KEYS)},
    }
    successful_missing_one = {
        "success": True,
        "env_steps": 20,
        "inference_chunks": 2,
        **{key: 10.0 for key in FORCE_METRIC_KEYS},
    }
    successful_missing_one["ap_max_meas"] = None
    failed = {
        "success": False,
        "env_steps": 30,
        "inference_chunks": 3,
        **{key: 1000.0 for key in FORCE_METRIC_KEYS},
    }

    aggregates, counts = aggregate_success_force_metrics(
        [successful, successful_missing_one, failed]
    )
    steps = summarize_step_statistics([successful, successful_missing_one, failed])

    assert aggregates["squeeze_avg_pred"] == pytest.approx(5.5)
    assert counts["squeeze_avg_pred"] == 2
    assert aggregates["ap_max_meas"] == pytest.approx(8.0)
    assert counts["ap_max_meas"] == 1
    assert steps["all"]["env_steps_total"] == 61
    assert steps["successful"]["episodes"] == 2
    assert steps["failed"]["inference_chunks_total"] == 3


def test_episode_validation_detects_duplicate_missing_partial_and_illegal_steps():
    episodes = [
        {
            "experiment_index": 0,
            "success": True,
            "env_steps": 11,
            "inference_chunks": 2,
            "force_status": "complete",
        },
        {
            "experiment_index": 0,
            "success": False,
            "env_steps": 21,
            "inference_chunks": 2,
            "force_status": "partial",
        },
    ]

    warnings = validate_episode_records(
        episodes,
        expected_count=2,
        replan_steps=10,
        max_inference_chunks=2,
    )

    assert any("duplicate experiment_index" in warning for warning in warnings)
    assert any("do not cover" in warning for warning in warnings)
    assert any("outside" in warning for warning in warnings)
    assert any("partial force coverage" in warning for warning in warnings)
