# Firm damage + explicit fixed friction 0.5

This profile extends `firm_damage_from_rlinf_sft_20k` for Firm Tasks
`0,1,2,3,5,6,7,8,9`.

- Both Franka finger bodies use static/dynamic friction `0.5/0.5`.
- Each Task's `obj_of_interest` uses static/dynamic friction `0.5/0.5`.
- Task 5 retains reset-time mass randomization in `[0.08, 0.12] kg`.
- Object-damage thresholds and `consecutive_frames=4` are unchanged.
- Task 4 remains unchanged and is not part of this Firm evaluation subset.

The fixed value deliberately matches the previous IsaacLab default material.
This formal profile therefore exercises the explicit Tabero friction path without
introducing an arbitrary new perturbation magnitude. Uniform randomization is
validated separately by a runtime smoke.

Direct Tabero evaluation:

```bash
python Tabero/scripts/tools/run_task_evaluations.py \
  --config-path /data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/firm_damage_fixed_friction_05_from_rlinf_sft_20k \
  ...
```

RLinf uses the existing YAML-only handoff:

```yaml
env:
  train:
    init_params:
      libero_config_dir: /data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/firm_damage_fixed_friction_05_from_rlinf_sft_20k
  eval:
    init_params:
      libero_config_dir: /data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/firm_damage_fixed_friction_05_from_rlinf_sft_20k
```

No RLinf source-code change is required.
