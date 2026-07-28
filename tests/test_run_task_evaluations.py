import ast
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools import run_task_evaluations as rte


def test_libero_path_under_tabero_checkout_does_not_auto_enable_tabero_subset():
    hdf5_folder = Path("/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/assembled_hdf5")

    assert not rte._should_auto_use_tabero_tasks(hdf5_folder, "")


def test_tabero_dataset_path_auto_enables_tabero_subset():
    hdf5_folder = Path("/datasets/tabero_force/replayed_demos")

    assert rte._should_auto_use_tabero_tasks(hdf5_folder, "")


def test_openpi_camera_names_cli_accepts_multiple_values():
    source = (ROOT / "benchmarks/openpi/openpi_inference_client.py").read_text()
    module = ast.parse(source)
    annotation = None

    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "OpenpiClientArguments":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id == "camera_names":
                        annotation = ast.unparse(item.annotation)

    assert annotation == "tuple[str, ...]"


def test_openpi_osc_actions_are_sent_as_7d_actions():
    source = (ROOT / "benchmarks/openpi/openpi_inference_client.py").read_text()

    osc_branch_start = source.find('elif args.control_mode == "osc":')
    assert osc_branch_start != -1

    next_branch_start = source.find('elif args.control_mode == "binary":', osc_branch_start)
    assert next_branch_start != -1

    osc_branch = source[osc_branch_start:next_branch_start]
    assert "action_chunk[:, :7]" in osc_branch
    assert "axisangle2quat" not in osc_branch


def test_openpi_sim_launcher_args_are_passed_to_client_command():
    config = rte.EvaluationConfig(
        policy_model="openpi",
        sim_device="cuda:0",
        sim_multi_gpu=True,
        sim_kit_args="--/renderer/activeGpu=8",
    )

    cmd = rte.build_command(config, "libero_object", 0)

    assert "--sim_device" in cmd
    assert cmd[cmd.index("--sim_device") + 1] == "cuda:0"
    assert "--sim_multi_gpu" in cmd
    assert "--sim_kit_args" not in cmd
    assert "--sim_kit_args=--/renderer/activeGpu=8" in cmd


def test_dsrl_raw_image_flag_is_opt_in_for_openpi_client_command():
    default_cmd = rte.build_command(rte.EvaluationConfig(policy_model="openpi"), "libero_object", 0)
    enabled_cmd = rte.build_command(
        rte.EvaluationConfig(policy_model="openpi", send_dsrl_raw_image=True),
        "libero_object",
        0,
    )

    assert "--send-dsrl-raw-image" not in default_cmd
    assert enabled_cmd.count("--send-dsrl-raw-image") == 1


def test_openpi_client_passes_sim_launcher_args_to_app_launcher():
    source = (ROOT / "benchmarks/openpi/openpi_inference_client.py").read_text()
    module = ast.parse(source)
    fields = set()

    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "OpenpiClientArguments":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.add(item.target.id)

    assert "sim_device" in fields
    assert "sim_multi_gpu" in fields
    assert "sim_kit_args" in fields

    assert "multi_gpu=args.sim_multi_gpu" in source
    assert "kit_args=args.sim_kit_args" in source


def test_libero_horizon_policy_is_consistent_in_command_and_json(tmp_path):
    config = rte.EvaluationConfig(policy_model="openpi", max_inference_steps=80)

    assert rte.effective_max_inference_steps(config, "libero_10") == 50
    assert rte.effective_max_inference_steps(config, "libero_object") == 30

    command = rte.build_command(config, "libero_object", 0)
    assert command[command.index("--max_inference_steps") + 1] == "30"

    output_path = tmp_path / "result.json"
    result = {
        "task_suite": "libero_object",
        "task_id": 0,
        "task_name": "pick_up_object",
        "language_instruction": "pick up the object",
        "success_rate": 50.0,
        "successful_experiments": 1,
        "total_experiments": 2,
        "max_inference_steps": 30,
        "execution_time": 1.0,
        "status": "completed",
    }
    rte.save_success_rates_json([result], output_path, config)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["metadata"]["max_inference_steps_policy"]["libero_10"] == 50
    assert saved["metadata"]["max_inference_steps_policy"]["libero_object"] == 30
    assert saved["results"]["libero_object_task0"]["max_inference_steps"] == 30


def test_plain_prompt_evaluation_omits_adverbs_and_records_metadata(tmp_path):
    config = rte.EvaluationConfig(
        policy_model="openpi",
        prompt_adverb="",
        prompt_adverbs=(),
        prompt_seed=0,
    )

    command = rte.build_command(config, "libero_object", 6)

    assert "--prompt_adverb" not in command
    assert "--prompt_adverbs" not in command

    output_path = tmp_path / "result.json"
    result = {
        "task_suite": "libero_object",
        "task_id": 6,
        "task_name": "pick_up_the_butter_and_place_it_in_the_basket",
        "language_instruction": "pick up the butter and place it in the basket",
        "success_rate": 70.0,
        "successful_experiments": 7,
        "total_experiments": 10,
        "max_inference_steps": 30,
        "execution_time": 1.0,
        "status": "completed",
    }
    rte.save_success_rates_json([result], output_path, config)
    metadata = json.loads(output_path.read_text(encoding="utf-8"))["metadata"]

    assert metadata["prompt_mode"] == "plain"
    assert metadata["prompt_adverb"] == ""
    assert metadata["prompt_adverbs"] == []
    assert metadata["prompt_seed"] == 0


def test_zero_exit_without_complete_metrics_is_failed(monkeypatch, capsys):
    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO("CUDA error: invalid device ordinal\n")
            self.returncode = 0

        def wait(self, timeout):
            assert timeout == 3600

    monkeypatch.setattr(rte.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    config = rte.EvaluationConfig(policy_model="openpi", num_total_experiments=1)

    result = rte.run_single_evaluation(
        config,
        task_suite="libero_object",
        task_id=0,
        workspace_root=ROOT,
    )

    assert result["return_code"] == 0
    assert result["success_rate"] is None
    assert result["status"] == "failed"
    assert "TASK FAILED: libero_object - Task 0" in capsys.readouterr().out
