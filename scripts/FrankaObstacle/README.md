# Franka static-obstacle comparison

`static_obstacle_test.py` creates 25 planner-independent A-B problems and runs
VanillaRRT and FlowEBRRT against the same fixed cuRobo scene. The scene contains
one axis-aligned cuboid centered at `[0.50, 0.00, 0.50]` m with full dimensions
`[0.20, 0.25, 0.30]` m.

Each selected problem satisfies all of these conditions:

- the start and goal obey joint/velocity limits;
- the complete robot is clear of self-collision and the cuboid at both endpoints;
- a 65-state direct joint interpolation is self-collision-free without the box;
- that same direct interpolation collides when the box is present; and
- a sampled two-segment geometric route around the cuboid is collision-free.

This makes the obstacle material to every problem without selecting problems
using either planner's outcome. Geometric reachability does not guarantee that
the kinodynamic planners will find a trajectory within their time budget.

Run the complete fixed test from the repository root with the cuRobo environment:

```powershell
third_party\curobo\.venv-curobo-windows\Scripts\python.exe `
  scripts\FrankaObstacle\static_obstacle_test.py `
  --dataset C:\Users\sodan\Downloads\dataset200k.h5
```

The completed 2026-08-05 run is under
`results/franka_obstacle/static_cuboid_25_20260805`. With 30 seconds per
problem, VanillaRRT solved 0/25 and FlowEBRRT with `best_inference.pt` solved
1/25 (problem 14 in 15.43 seconds). All saved paths passed a cuboid-aware
MorphIt/cuRobo audit. The script now defaults to a normalized joint-position
goal radius of 0.40; this saved historical run used the earlier 0.25 radius.

That completed run used the then-active 79-sphere density-2 robot model. The
Franka runtime was subsequently switched to the 295-sphere
`panda_morphit_d10_c1000_p1000.yml` model. The saved results remain historical
79-sphere artifacts; a new run is required for 295-sphere obstacle results.

At a batch size of 256 on the RTX 2060, self-collision alone took 5.99
microseconds/configuration and self-plus-cuboid checking took 9.48
microseconds/configuration (105,458 configurations/second). Each RRT rollout
edge is submitted as one batch.

Visualize any result by specifying both the problem and algorithm:

```powershell
.venv\Scripts\python.exe `
  scripts\FrankaObstacle\visualize_static_obstacle.py `
  --problem-id 14 --planner flow
```

Use `--planner vanilla` for VanillaRRT. Add `--headless --save-image PATH` to
render a PNG without opening the PyBullet GUI. Failed trials visualize their
best partial path; successful trials visualize the solution path.

## Four-Franka shared-table scene

`four_franka_table_scene.py` creates a reusable PyBullet environment with a
1.80 x 0.90 x 0.68 m central cuboid table and four inward-facing Panda arms,
two on each long side. Robot bases are at x = +/-0.52 m, y = +/-0.74 m, and
z = 0.76 m. This gives 1.48 m between opposing bases, 1.04 m between bases on
the same side, and 0.29 m from each base center to the nearest table edge.

Each robot instance is bound to the same two-part geometry definition:

- PyBullet renders the 10 articulated link meshes from
  `assets/robots/panda/panda.urdf` and uses that URDF's collision meshes for
  visualization-time baseline checks.
- Planning uses the 295 MorphIt spheres from
  `assets/robots/panda/curobo/panda_morphit_d10_c1000_p1000.yml` for the same
  URDF and kinematic chain.

The scene generator checks that the URDF named by the cuRobo configuration is
exactly the URDF loaded into PyBullet and refuses to render a model with any
collision-sphere count other than 295. The scene therefore contains 1,180
planning spheres across four identical mesh robots. The JSON manifest records
the geometry files and sphere count for every robot.

Render the labeled metric-grid view and write the machine-readable scene
manifest from the repository root:

```powershell
third_party\curobo\.venv-curobo-windows\Scripts\python.exe `
  scripts\FrankaObstacle\four_franka_table_scene.py
```

The PNG and JSON manifest are written to
`results/franka_obstacle/four_franka_table_scene/`. Add `--gui` to inspect the
same scene interactively in PyBullet.

### Difficult four-arm problems

`create_four_franka_difficult_problems.py` defines three fixed four-arm
start/goal problems in the shared-table scene. Every endpoint is collision-free
under the active 295-sphere MorphIt/cuRobo self-collision model and PyBullet
table/inter-robot checks. Each problem is deliberately coordination-heavy: a
41-sample synchronized straight-line joint interpolation has inter-robot
collisions, while both endpoints remain valid.

```powershell
third_party\curobo\.venv-curobo-windows\Scripts\python.exe `
  scripts\FrankaObstacle\create_four_franka_difficult_problems.py
```

The output folder `results/franka_obstacle/four_franka_difficult_problems/`
contains three labeled start/collision-witness/goal images, a JSON validation
manifest, and a compressed NPZ problem set. The NPZ arrays `starts` and `goals`
have shape `(3, 4, 14)`: three problems, four robots in NW/NE/SW/SE order, and
seven joint positions followed by seven zero endpoint velocities.

### Using `prioritized_planning.py`

The orchestration in `mrmp_with_kite_extend/src/prioritized_planning.py` is
the right high-level algorithm: accept four ordered low-level planners, plan
robot 0, expose its time-indexed trajectory as a moving obstacle, then repeat
for robots 1 through 3. The priority order is the order of the `planners`
list. Once four compatible planners exist, the call is:

```python
solved, wall_time, total_cost = PrioritizedPlanning.plan_multi(
    planners=[northwest, northeast, southwest, southeast],
    planning_time=120.0,
    print_logs=True,
    dynamic_obstacle_adapter=adapter,
)
```

`franka_prioritized_adapter.py` now preserves that outer loop and replaces the
legacy single-radius dynamic obstacle with articulated GPU geometry:

1. Load one `(4, 14)` start/goal pair from
   `four_franka_difficult_problems.npz` and create four `FrankaPanda` agents.
2. Give every agent the same URDF and 295-sphere model, plus its fixed
   world-from-base transform from `ROBOT_LAYOUT`.
3. Include the table, pedestal, and mount cuboids in each arm's static cuRobo
   scene, expressed in that arm's base frame.
4. After a higher-priority arm is planned, evaluate FK for its dense trajectory
   and cache a world-frame array shaped `(time, 295, 4)`, where the final
   coordinate stores sphere radius.
5. During each candidate rollout for a lower-priority arm, evaluate all of its
   295 world-frame spheres in one GPU batch and test sphere-sphere overlap
   against the cached higher-priority trajectories at matching time indices.
   When one trajectory ends, hold its final configuration indefinitely.
6. Apply the same comparison to the candidate's stationary goal tail so a
   robot is not accepted at a goal that a higher-priority arm occupies later.
7. Run `PrioritizedPlanning.plan_multi` with its optional adapter hook and then
   replay all four paths synchronously through an independent cuRobo/GPU audit.

The 295-by-295 pair tests should remain on the GPU and be batched over rollout
waypoints and higher-priority robots. PyBullet is used only by
`visualize_four_franka_prioritized.py` to render the saved paths to MP4.

Run all three scenarios with both base planners:

```powershell
third_party\curobo\.venv-curobo-windows\Scripts\python.exe `
  scripts\FrankaObstacle\prioritized_four_franka.py `
  --problem-ids 0,1,2 --algorithms vanilla,flow `
  --planning-time 120 --state-bank-size 100000
```

Audit a completed run without PyBullet:

```powershell
third_party\curobo\.venv-curobo-windows\Scripts\python.exe `
  scripts\FrankaObstacle\audit_four_franka_prioritized.py `
  --run-dir results\franka_obstacle\four_franka_prioritized\RUN_NAME
```

Render its videos afterward:

```powershell
third_party\curobo\.venv-curobo-windows\Scripts\python.exe `
  scripts\FrankaObstacle\visualize_four_franka_prioritized.py `
  --run-dir results\franka_obstacle\four_franka_prioritized\RUN_NAME
```

The final 2026-08-07 run is `run_20260807_final`. With the shared 120-second
budget and order NW/NE/SW/SE, FlowEBRRT solved 3/3 scenarios and VanillaRRT
solved 1/3. All six saved solution/partial-path sets passed the synchronized
cuRobo replay audit. Complete generation-level statistics are stored in
`combined_generations.csv`; each result also contains `generations.csv`,
`summary.json`, `trajectories.npz`, and `curobo_audit.json`.
