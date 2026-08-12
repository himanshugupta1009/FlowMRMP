# MRMP upstream sync and Franka agent relocation (2026-08-07)

## Upstream source

- Repository: `https://github.com/himanshugupta1009/mrmp_with_kite_extend`
- Upstream default branch: `master`
- Mirrored commit: `5ac5aa2b4086fd0291402b33feed871fa6559f97`
- Commit subject: `Align ablation plots with current results and budgets`

`mrmp_with_kite_extend` is a vendored directory tracked by the main FrankaFM
repository. It was not a Git submodule or an independent nested checkout. The
upstream `master` worktree was therefore mirrored into that directory while
excluding the upstream `.git` directory. The populated
`mrmp_with_kite_extend/pybullet-planning` directory was preserved; its gitlink
commit is `49385a1e561335a5ba7eba0907443924bf3f44d7`, matching upstream.

## Franka-owned files moved out of the vendored tree

- `mrmp_with_kite_extend/src/Agents/FrankaPanda.py` moved to
  `scripts/FrankaPanda.py`.
- `mrmp_with_kite_extend/src/Agents/redirect_stream.py` moved to
  `scripts/franka_redirect_stream.py`.
- Both old files were removed from the vendored MRMP directory. They are not
  present in upstream `master` either.
- `scripts/FrankaPanda.py` now imports its helper from the main scripts tree and
  resolves its default cuRobo configuration from the FrankaFM repository root.

The active default remains
`assets/robots/panda/curobo/panda_morphit_d10_c1000_p1000.yml`, containing 295
MorphIt collision spheres.

## Import and provenance updates

The following callers now add the main `scripts` directory to `sys.path` where
needed and import `FrankaPanda` from there instead of from `src/Agents`:

- `scripts/audit_franka_edge_bundle_dataset.py`
- `scripts/create_franka_edge_bundle_dataset.py`
- `Tests/test_franka_rrt_planning_changes.py`
- `scripts/FrankaObstacle/create_four_franka_difficult_problems.py`
- `scripts/FrankaObstacle/static_obstacle_test.py`
- `scripts/FrankaObstacle/visualize_static_obstacle.py`
- `scripts/FrankaTesting/audit_franka_reachable_problem_set.py`
- `scripts/FrankaTesting/audit_franka_rrt_results.py`
- `scripts/FrankaTesting/benchmark_franka_collision_backends.py`
- `scripts/FrankaTesting/benchmark_franka_flow_eb_rrt.py`
- `scripts/FrankaTesting/benchmark_franka_vanilla_rrt.py`
- `scripts/FrankaTesting/create_franka_reachable_problem_set.py`
- `scripts/FrankaTesting/evaluate_franka_edge_flow_matching.py`
- `scripts/FrankaTesting/repair_franka_problem_set_for_active_collision.py`
- `scripts/FrankaTesting/visualize_franka_rrt.py`

Source-hash metadata paths were updated in the RRT audit, FlowEBRRT benchmark,
and edge-flow evaluation scripts so recorded provenance points to the relocated
agent.

## Local compatibility merge in current upstream RRT

The latest upstream `src/rrt.py` only selected a dynamic collision routine for
2D or 3D agents. Franka uses a normalized 14D nearest-neighbor state and
intentionally disables moving-agent tail checks for its single-arm benchmark,
so the unmodified upstream constructor rejected it. A narrow compatibility
merge was applied to `mrmp_with_kite_extend/src/rrt.py` while retaining the new
upstream goal-parking and tree-snapshot behavior:

- Honor an agent's `disable_dynamic_collision_check` flag.
- Use an agent-provided distance-metric projection and distance function.
- Sample rollout durations as integer integration steps.
- Truncate a rollout at the first goal-reaching waypoint before validity
  checking the accepted prefix.
- Reconstruct dense paths from stored waypoint counts rather than rounded time.
- Record planning wall time and iteration count.
- Recognize a start state already inside the goal and verify every reported
  goal result.

This merge is required by both VanillaRRT and FlowEBRRT; it does not alter the
Franka URDF, kinematics, MorphIt sphere configuration, datasets, checkpoints,
or result files.

## Verification

- Upstream mirror audit before the compatibility merge: 382 tracked regular
  files checked, 0 missing, 0 different, and 0 unexpected extras (excluding
  the preserved `pybullet-planning` gitlink directory).
- Focused regression suite: 7 passed.
  - `Tests/test_franka_rrt_planning_changes.py`
  - `Tests/test_franka_edge_bundle_validity_filter.py`
- Python byte-compilation passed for `scripts`, `mrmp_with_kite_extend/src`,
  and `Tests`.
- `pip check`: no broken requirements in the cuRobo environment.
- `--help` smoke tests passed for the VanillaRRT benchmark, FlowEBRRT benchmark,
  and static-obstacle comparison entry points.
- Relocated agent import resolves the intended default configuration and the
  configuration contains 295 collision spheres.
