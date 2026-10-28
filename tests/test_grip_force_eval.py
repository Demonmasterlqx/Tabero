"""CPU checks for evaluation conventions, including the production physics-step EMA."""

import argparse
import ast
import importlib.util
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch
import tyro

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common.episode_metrics import aggregate_success_force_metrics, summarize_episode_force_metrics
from benchmarks.common.metrics import (
    compute_contact_force_metrics_from_13d,
    compute_contact_force_series_from_lr_forces,
    grip_force_metric_metadata,
)
from scripts.tools import run_data_evaluations as rde
from scripts.tools import run_task_evaluations as rte

CONTROLLER = ROOT / "source/tac_manip/tac_manip/tasks/manipulation/libero/mdp/force_position_action.py"
CLIENT = ROOT / "benchmarks/openpi/openpi_inference_client.py"


@pytest.mark.parametrize("path", [CLIENT, ROOT / "scripts/tools/replay_demos.py", ROOT / "scripts/tools/replay_demos_with_camera.py"])
def test_evaluation_entrypoints_resolve_controller_from_this_checkout(path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    # Execute the entrypoint's path bootstrap without loading Isaac/Kit.
    nodes = [node for node in ast.parse(path.read_text()).body if (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"project_root", "_PROJECT_ROOT"}
            for target in node.targets
        ) or isinstance(node, ast.If) and "sys.path" in ast.unparse(node.test) or
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "sys.path.insert"
    )]
    namespace = dict(sys=sys, Path=Path, __file__=str(path))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    assert Path(importlib.util.find_spec("tac_manip").origin).resolve() == (
        ROOT / "source/tac_manip/tac_manip/__init__.py"
    )


def test_sum_normal_force_handles_asymmetry_signs_and_zero_without_changing_applied_force():
    left = np.array([[0, 0, 2], [0, 0, -10], [0, 0, 0], [0, 0, 0], [100, 0, 2]], dtype=float)
    right = np.array([[0, 0, -10], [0, 0, -10], [0, 0, 8], [0, 0, 0], [0, 100, 10]], dtype=float)
    old = compute_contact_force_series_from_lr_forces(left, right, old_eval=True)
    new = compute_contact_force_series_from_lr_forces(left, right, old_eval=False)
    np.testing.assert_array_equal(old.squeeze, [4, 20, 0, 0, 4])
    np.testing.assert_array_equal(new.squeeze, [12, 20, 8, 0, 12])
    np.testing.assert_array_equal(new.external, old.external)
    np.testing.assert_array_equal(new.contact_mask, old.contact_mask)
    # Existing non-evaluation callers keep their historical default.
    np.testing.assert_array_equal(compute_contact_force_series_from_lr_forces(left, right).squeeze, old.squeeze)


@pytest.mark.parametrize("old_eval,top5", [(False, 50.0), (True, 40.0)])
def test_episode_and_success_aggregation_preserve_population_and_top5(old_eval, top5):
    # Two zero frames, 38 (2,10) frames and two distinct high-force frames.
    # Top 5% of 40 positive frames includes both high frames, not just the max.
    actions = np.zeros((42, 13), dtype=np.float32)
    actions[2:40, 9] = 2
    actions[2:40, 12] = -10
    actions[40:, 9] = [10, 30]
    actions[40:, 12] = [-30, -30]
    grip = np.array([0, 0] + ([4] * 38 + [20, 60] if old_eval else [12] * 38 + [40, 60]))
    summary = summarize_episode_force_metrics(
        force_mode="full", env_steps=42, old_eval=old_eval,
        actions_13d=actions, squeeze_pred_values=grip, squeeze_meas_values=grip,
        ap_pred_values=np.zeros(42), ap_meas_values=np.zeros(42),
        fL_meas_values=actions[:, 7:10], fR_meas_values=actions[:, 10:13],
    )
    # All-frame averages retain the two zero-force frames.
    expected_mean = (232 if old_eval else 556) / 42
    assert summary["squeeze_avg_pred"] == pytest.approx(expected_mean)
    assert summary["squeeze_avg_meas"] == pytest.approx(expected_mean)
    assert summary["squeeze_max_pred"] == pytest.approx(top5)
    assert summary["squeeze_max_meas"] == pytest.approx(top5)
    assert summary["force_status"] == "complete"
    assert summary["force_samples"]["predicted_contact_steps"] == 40
    # Shared helper means use contact frames; evaluator AG uses the whole episode.
    contact = compute_contact_force_metrics_from_13d(actions, old_eval=old_eval)
    assert contact.squeeze_mean == pytest.approx((232 if old_eval else 556) / 40)
    aggregates, counts = aggregate_success_force_metrics([
        {**summary, "success": True},
        {"success": True, "squeeze_avg_pred": 0.0},
        {"success": False, "squeeze_avg_pred": 10000.0},
    ])
    assert aggregates["squeeze_avg_pred"] == pytest.approx(expected_mean / 2)
    assert counts["squeeze_avg_pred"] == 2
    assert counts["squeeze_avg_meas"] == 1


@pytest.mark.parametrize("old_eval,grip,peak", [(False, [0, 12, 20], 20), (True, [0, 4, 20], 20)])
def test_binary_mode_keeps_measured_only_compatibility_slots(old_eval, grip, peak):
    summary = summarize_episode_force_metrics(
        force_mode="measured_only", env_steps=3, old_eval=old_eval,
        squeeze_pred_values=[0, 0, 0], squeeze_meas_values=grip,
        ap_pred_values=[0, 0, 0], ap_meas_values=[0, 8, 0],
        fL_meas_values=[[0, 0, 0], [0, 0, 2], [0, 0, 10]],
        fR_meas_values=[[0, 0, 0], [0, 0, -10], [0, 0, -10]],
    )
    assert summary["squeeze_avg_pred"] == 0.0
    assert summary["squeeze_avg_meas"] == pytest.approx(sum(grip) / 3)
    assert summary["squeeze_max_pred"] == summary["squeeze_max_meas"] == peak
    assert summary["ap_avg_pred"] == summary["ap_avg_meas"] == 4.0
    assert summary["force_status"] == "complete"


def _load_controller():
    """Execute actual controller methods with CPU tensors and fake hardware."""
    tree = ast.parse(CONTROLLER.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ForcePositionAction")
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {
        "apply_actions", "reset", "_split_squeeze_and_applied_from_lr_local",
    }]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace = {
        "torch": torch,
        "contact_force_in_gripper_frame": lambda env, **kwargs: env.force,
        "math_utils": SimpleNamespace(
            quat_apply=lambda quat, value: value,
            quat_apply_inverse=lambda quat, value: value,
            quat_from_angle_axis=lambda angle, axis: torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ),
    }
    exec(compile(ast.fix_missing_locations(module), str(CONTROLLER), "exec"), namespace)
    return namespace["ForcePositionAction"]()


def _cpu_controller(alpha):
    controller = _load_controller()
    controller.num_envs = 1
    controller._device = "cpu"
    controller.cfg = SimpleNamespace(
        contact_sensor_name="contact_gripper", history_length=1,
        meas_force_filter_alpha=alpha, squeeze_ff_k_load_z=0.0,
        target_contact_squeeze_enabled=False, squeeze_kp=0.001,
        squeeze_deadzone=0.0, pos_kp=0.0,
        target_contact_single_finger_normal_force_n=0.0,
    )
    controller._env = SimpleNamespace(cfg=SimpleNamespace(gripper_open_val=0.04))
    controller._contact_sensor = object()
    controller._target_contact_sensor = None
    controller._fL_target_local = torch.tensor([[0.0, 0.0, 2.0]])
    controller._fR_target_local = torch.tensor([[0.0, 0.0, -10.0]])
    controller._f_sq_meas_ema = torch.zeros(1)
    controller._f_grip_sum_meas_ema = torch.zeros(1)
    controller._f_sq_meas_ema_initialized = False
    controller._left_frame = SimpleNamespace(data=SimpleNamespace(target_quat_w=torch.zeros(1, 1, 4)))
    controller._robot = SimpleNamespace(
        data=SimpleNamespace(root_quat_w=torch.zeros(1, 4), joint_pos=torch.zeros(1, 2)),
        set_joint_position_target=lambda value, **kwargs: None,
    )
    controller._ik_term = SimpleNamespace(
        action_dim=7, process_actions=lambda actions: None,
        apply_actions=lambda: None, reset=lambda **kwargs: None,
    )
    controller._gripper_joint_ids = [0, 1]
    controller._gripper_abs_cmd = torch.tensor([[0.02]])
    controller._eef_pos_cmd = torch.zeros(1, 3)
    controller._eef_aa_cmd = torch.zeros(1, 3)
    controller._raw_actions = torch.zeros(1, 13)
    controller._processed_actions = torch.zeros(1, 13)
    controller._target_contact_detected = torch.zeros(1, dtype=torch.bool)
    controller._target_contact_latched = torch.zeros(1, dtype=torch.bool)
    controller._target_contact_force_norm = torch.zeros(1)
    controller._target_contact_activation_step = torch.full((1,), -1)
    controller._environment_step_count = torch.zeros(1, dtype=torch.long)
    return controller


@pytest.mark.parametrize("alpha,expected_old,expected_new", [(0.2, 5.76, 12.48), (1.0, 0.0, 0.0)])
def test_production_controller_ema_uses_every_physics_step_and_preserves_commands(alpha, expected_old, expected_new):
    controller = _cpu_controller(alpha)
    # Three physics substeps in one control step: samples must not be collapsed
    # into a single EMA update at the final zero-force observation.
    for left, right in [(2, -10), (20, -10), (0, 0)]:
        controller._env.force = torch.tensor([[[[0.0, 0.0, left], [0.0, 0.0, right]]]])
        controller.apply_actions()
    debug = controller._debug
    assert debug["f_grip_sum_pred"].item() == pytest.approx(12)
    assert debug["f_sq_pred"].item() == pytest.approx(4)
    assert debug["f_sq_meas"].item() == pytest.approx(expected_old)
    assert debug["f_grip_sum_meas"].item() == pytest.approx(expected_new)
    # Control still uses the old squeeze error, even when new evaluation is read.
    assert debug["d_cmd"].item() == pytest.approx(0.02 - 0.001 * 0.5 * (4 - expected_old))
    controller.reset()
    controller._env.force = torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, -9.0]]]])
    controller.apply_actions()
    assert controller._debug["f_sq_meas"].item() == pytest.approx(2)
    assert controller._debug["f_grip_sum_meas"].item() == pytest.approx(10)


def _client_arguments():
    namespace = dict(dataclass=dataclass, Path=Path, os=os, Optional=Optional,
                     project_root=ROOT, __file__=str(CLIENT), __name__=__name__)
    nodes = []
    for path, name in [(ROOT / "benchmarks/common/closedloop_policy_inference.py", "ClosedLoopArguments"),
                       (CLIENT, "OpenpiClientArguments")]:
        nodes.append(next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == name))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CLIENT), "exec", dont_inherit=True), namespace)
    return namespace["OpenpiClientArguments"]


@pytest.mark.parametrize("old_eval", [False, True])
def test_cli_defaults_and_exact_old_eval_flag_reach_policy_client(old_eval):
    argv = ["--old_eval"] if old_eval else []
    config = tyro.cli(rte.EvaluationConfig, args=argv)
    command = rte.build_command(config, "libero_object", 0)
    client_args = tyro.cli(_client_arguments(), args=command[2:])
    assert config.old_eval is old_eval
    assert client_args.old_eval is old_eval
    assert tyro.cli(rde.DataReplayEvalConfig, args=argv).old_eval is old_eval


@pytest.mark.parametrize("script", ["replay_demos.py", "replay_demos_with_camera.py"])
@pytest.mark.parametrize("old_eval", [False, True])
def test_replay_parsers_accept_exact_flag(script, old_eval):
    # Build production argparse declarations without importing Isaac or launching it.
    path = ROOT / "scripts/tools" / script
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parser" for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "parser":
                nodes.append(node)
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    args = namespace["parser"].parse_args(["--old_eval"] if old_eval else [])
    assert args.old_eval is old_eval


@pytest.mark.parametrize("old_eval", [False, True])
def test_data_evaluator_passes_flag_to_replay_subprocess(old_eval, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/tools"))
    monkeypatch.setenv("REPLAYED_DEMOS_DIR", str(tmp_path))
    (tmp_path / "libero_object_task0_fixture_demo.hdf5").touch()
    commands = []

    def fake_popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            stdout=io.StringIO("Successfully replayed 1 episode out of 1 demos.\nFinished replaying 1 episode.\n"),
            wait=lambda: 0,
        )

    monkeypatch.setattr(rde.subprocess, "Popen", fake_popen)
    result = rde.run_single_replay_evaluation(
        rde.DataReplayEvalConfig(old_eval=old_eval, task="Isaac-Libero-Franka-Hybrid-Tactile-v0"),
        "libero_object", 0, ROOT,
    )
    assert ("--old_eval" in commands[0]) is old_eval
    assert result["old_eval"] is old_eval
    assert result["success_rate"] == 1.0


@pytest.mark.parametrize("old_eval", [False, True])
def test_policy_and_replay_outputs_record_metric_convention(old_eval, tmp_path):
    metadata = grip_force_metric_metadata(old_eval)
    assert metadata["grip_force_definition"] == (
        "2 * min(abs(fL_z), abs(fR_z))" if old_eval else "abs(fL_z) + abs(fR_z)"
    )
    result = dict(task_suite="libero_object", task_id=0, task_name="task", language_instruction="pick up",
                  success_rate=100.0, successful_experiments=1, total_experiments=1,
                  execution_time=0.1, status="completed")
    config = rte.EvaluationConfig(old_eval=old_eval)
    rte.save_success_rates_json([result], tmp_path / "policy.json", config)
    rte.save_success_rates_txt([result], tmp_path / "policy.txt", config)
    saved = json.loads((tmp_path / "policy.json").read_text())
    for key, value in metadata.items():
        assert saved["metadata"][key] == value
    assert metadata["grip_force_definition"] in (tmp_path / "policy.txt").read_text()
    rde.save_results([result], rde.DataReplayEvalConfig(old_eval=old_eval, output_dir=tmp_path))
    replay_json = json.loads(next(tmp_path.glob("replay_eval_*.json")).read_text())
    for key, value in metadata.items():
        assert replay_json["config"][key] == value
    assert metadata["grip_force_definition"] in next(tmp_path.glob("replay_eval_*.txt")).read_text()
