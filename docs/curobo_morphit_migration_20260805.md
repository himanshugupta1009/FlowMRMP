# Franka MorphIt/cuRobo migration record

Date: 2026-08-05  
Workspace: `C:\Projects\FrankaFM`  
Status: runtime replacement complete and validated on the local RTX 2060

## Outcome

The Franka planner's runtime validity checker now uses a MorphIt sphere model
through cuRobo's CUDA collision API. Both VanillaRRT and FlowEBRRT call the same
`FrankaPanda.is_new_node_valid` method; that method checks joint and velocity
limits and submits the entire candidate rollout edge to cuRobo in one batch.
PyBullet remains only as an explicit reference checker and as the GUI renderer
used by `visualize_franka_rrt.py`.

The active robot model was switched from the original 79-sphere density-2 fit
to the 295-sphere fit on 2026-08-05:

- cuRobo config: `assets/robots/panda/curobo/panda_morphit_d10_c1000_p1000.yml`
- 295 fitted spheres across 11 collision links
- URDF: `assets/robots/panda/panda.urdf`
- URDF SHA-256: `32e9be7d8957f26136274578993a1c34de48badcc9f051ad8c9d20b32031829f`
- SRDF: `assets/robots/panda/panda.srdf`
- SRDF SHA-256: `95aae52d0ba6ae278b747b63fe880ca72e97e4b72f5404af3fc2cc7882e1f197`
- articulated joint order: `panda_joint1` through `panda_joint7`
- fingers: fixed, as defined by the project URDF; movable gripper support was
  intentionally excluded from this step
- tool frame: `panda_grasptarget`
- self-collision exclusions: exactly the 34 pairs in the project SRDF
- self-collision buffer: 0 m

MorphIt was told to fit the URDF collision meshes, not the visual meshes. No
replacement URDF or alternate kinematic chain was introduced.

## Where collision and validity checking occurs

The RRT classes do not contain robot collision geometry. They call an injected
agent validity method:

1. Vanilla RRT and FlowEBRRT generate a candidate rollout edge.
2. Both call `FrankaPanda.is_new_node_valid`.
3. `FrankaPanda.is_new_node_valid` applies the 14D state limits and calls
   `FrankaCuroboCollisionChecker.collision_free_mask` once for all valid-limit
   joint configurations in the edge.
4. cuRobo performs forward kinematics, sphere self-collision, and—when a
   `scene_model` is supplied—robot-to-world obstacle collision checks.
5. The first invalid waypoint and rejection type are returned to the RRT.

Therefore the validity decision is used *inside* both RRT extension loops, but
the Franka-specific collision implementation lives in
`scripts/FrankaPanda.py`.

## Dependency setup

cuRobo was cloned into the ignored directory `third_party/curobo` at:

- repository: `https://github.com/NVlabs/curobo.git`
- commit: `8e734f3ced1df898990bcd92de40abce475907db`
- reported version: `0.8.0.post1.dev42`

The isolated environment is
`third_party/curobo/.venv-curobo-windows` and contains:

- Python 3.12
- PyTorch `2.8.0+cu126`
- CUDA runtime 12.6
- cuRobo editable install with its `cu12` extras
- `viser==1.0.30`
- `tifffile==2025.5.10`
- the packages in `requirements-franka.txt`
- PyBullet 3.2.7 for reference measurements and visualization
- pandas and pytest for benchmark/test tooling

`pip check` reports no broken requirements. `viser` was upgraded from the
resolver's initial 0.2.23 choice because that version required a `liblzfse`
wheel unavailable on Windows. `tifffile` was pinned because the newest release
required NumPy 2.1 while this project pins NumPy 2.0.2.

Recreate the environment with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_franka_curobo.ps1
```

An attempt was made to enable WSL and Virtual Machine Platform without a
reboot. Windows required an elevated feature change and WSL remained disabled.
No verified Windows-feature change was left behind. Native Windows cuRobo was
then tested and its CUDA kernels executed successfully, so WSL was not needed.

## Kinematics, obstacle, and self-collision validation

Kinematics were compared between the exact URDF and cuRobo for 256 random
configurations and 12 non-base link/tool poses per configuration (3,072 pose
comparisons):

| Metric | Mean | Maximum |
|---|---:|---:|
| Translation error | 5.11e-8 m | 2.39e-7 m |
| Rotation error | 1.10e-7 rad | 6.82e-7 rad |

The cuRobo world-collision API was exercised with three scenes: an empty world
was valid, a distant cuboid was valid, and an 8 cm cuboid centered on the tool
was rejected. This confirms that the same checker can include static obstacles
by passing a cuRobo `scene_model`; the current 100-problem benchmark remains an
empty-world self-collision benchmark.

The original density-2 model was compared with exact-mesh PyBullet on a fresh
100,000-state holdout:

| Result | Count/rate |
|---|---:|
| PyBullet collision states | 13,656 |
| True positives | 13,653 |
| False negatives | 3 |
| Collision recall | 99.978% |
| False positives | 34,070 |
| Overall agreement | 65.927% |

This is an important historical limitation: the 79-sphere approximation is much more
conservative than the exact meshes. It nearly always catches a PyBullet
collision but rejects many PyBullet-valid configurations. Higher-density,
weighted, global-margin, and targeted-padding variants were tested. None gave
both the selected model's collision recall and acceptable benchmark-state
coverage, so they were retained as experimental artifacts but not activated.
The rare false negatives mean this sphere model is not a mathematical safety
certificate and should not be the sole safeguard on physical hardware.

## Collision timing on this machine

Hardware: NVIDIA GeForce RTX 2060, compute capability 7.5, 6 GB VRAM.

| Checker / batch | Microseconds per configuration | Configurations/s |
|---|---:|---:|
| PyBullet exact mesh, scalar | 32.91 | 30,382 |
| cuRobo density-2, batch 1 | 1,913.86 | 523 |
| cuRobo density-2, batch 16 | 88.14 | 11,345 |
| cuRobo density-2, batch 64 | 22.10 | 45,254 |
| cuRobo density-2, batch 256 | 5.59 | 178,945 |
| cuRobo density-2, batch 4,096 | 0.375 | 2,663,407 |

These timings describe the original 79-sphere density-2 model, not the current
295-sphere runtime model. cuRobo is slower when called for one configuration at a time because CUDA
launch overhead dominates. It becomes faster at a 64-state batch and is about
5.9x faster per configuration at a 256-state batch. This is why edge validation
was changed to one batched call.

## 100-problem planner evaluation

The old PyBullet-valid problem file could not be reused fairly: MorphIt rejected
at least one witness state in 59/100 problems, and only 45/100 problems had both
start and goal valid. The original file was preserved. A new, independently
selected MorphIt-valid set was created at
`data/benchmarks/franka_reachable_morphit_ab_100.npz` (SHA-256
`94ac4e28b94f94a3ed255bf05404894ca7326895c7fc21b4e1d6de34ea16c1b2`).
It contains 40 easy, 40 medium, and 20 hard problems from 100 unique source
trajectories; every stored witness state passes limits and the active checker.

Both planners used the same set, seed 20260721, 30 seconds/problem, normalized
joint-position goal radius 0.25, empty world, and one problem at a time.

| Planner | Successes | Success rate | Mean time, all | Median time, all | Mean time, successes |
|---|---:|---:|---:|---:|---:|
| VanillaRRT | 78/100 | 78% | 9.18 s | 1.70 s | 3.30 s |
| FlowEBRRT + `best_inference.pt` | 70/100 | 70% | 10.18 s | 0.204 s | 1.68 s |

Paired outcomes were 62 both succeeded, 8 Flow-only, 16 Vanilla-only, and 14
neither. All 78 Vanilla successes and all 70 Flow successes passed the saved
path validity audit. The learned planner is substantially quicker when it
succeeds, but Vanilla solved eight more problems at this fixed timeout.

## Complete change ledger

### Edited repository files

- `.gitignore`: ignores the local cuRobo checkout and its virtual environment.
- `scripts/FrankaPanda.py`: moved the project-owned Franka adapter out of the
  independently maintained MRMP source tree; it contains the cuRobo checker,
  kept PyBullet as an explicit reference class, made the legacy checker name
  resolve to cuRobo, and batched full-edge validity calls.
- `scripts/FrankaTesting/create_franka_reachable_problem_set.py`: batches
  witness collision checks when the backend exposes `collision_free_mask`.
- `scripts/FrankaTesting/visualize_franka_rrt.py`: explicitly selects the
  PyBullet checker because its GUI/body API is still the renderer.
- `scripts/FrankaTesting/benchmark_franka_vanilla_rrt.py`: corrected runtime
  metadata to describe batched MorphIt/cuRobo validation.
- `scripts/FrankaTesting/benchmark_franka_flow_eb_rrt.py`: corrected runtime
  and collision-geometry metadata.
- `scripts/FrankaTesting/audit_franka_rrt_results.py`: reports the active
  collision backend and records cuRobo/PyTorch dependency versions.
- `scripts/FrankaTesting/compare_franka_rrt_results.py`: corrected comparison
  notes to describe the shared MorphIt model.

### New repository files

- `scripts/setup_franka_curobo.ps1`: reproducible pinned environment setup.
- `scripts/FrankaTesting/build_franka_curobo_model.py`: exact-asset MorphIt
  fitting plus model/kinematic report generation.
- `scripts/FrankaTesting/benchmark_franka_collision_backends.py`: PyBullet
  versus cuRobo correctness, kinematics, obstacle, and timing benchmark.
- `scripts/FrankaTesting/create_franka_curobo_margin_variants.py`: experimental
  global self-collision-margin variants.
- `scripts/FrankaTesting/create_franka_curobo_safety_variant.py`: experimental
  targeted sphere-padding variant.
- `third_party/README.md`: pinned third-party provenance and setup command.
- `docs/curobo_morphit_migration_20260805.md`: this record.

### New generated robot-model files

The current active files are `panda_morphit_d10_c1000_p1000.yml` and
`panda_morphit_d10_c1000_p1000_build.json`. The original selected files were
`panda_morphit_density2.yml` and `panda_morphit_density2_build.json`. The same
directory also contains the following explicitly retained experiments:

- base density models: `panda_morphit.yml`, `panda_morphit_density4.yml`, and
  their `_build.json` reports
- weighted models: `panda_morphit_d2_c1000_p100`,
  `panda_morphit_d2_c1000_p1000`, `panda_morphit_d4_c1000_p100`,
  `panda_morphit_d4_c100_p100`, `panda_morphit_d10_c1000_p100`,
  `panda_morphit_d10_c1000_p300`, `panda_morphit_d10_c1000_p500`,
  `panda_morphit_d10_c1000_p1000`, `panda_morphit_d20_c1000_p300`, and
  `panda_morphit_d20_c1000_p1000`, each with YAML and build JSON
- global-margin YAMLs for 1, 2, 3, 5, 7.5, 10, and 15 mm, plus
  `panda_morphit_self_margin_variants.json`
- `panda_morphit_density2_safety_padded.yml` and its JSON report

All are under `assets/robots/panda/curobo/`; the runtime now references only the
295-sphere `d10_c1000_p1000` model.

### New benchmark data and results

- `data/benchmarks/franka_reachable_morphit_ab_100.npz`
- `data/benchmarks/franka_reachable_morphit_ab_100.audit.json`
- `results/franka_collision_backends/`: model comparisons, holdouts,
  calibration sweeps, kinematic/obstacle tests, and the old-problem audit
- `results/franka_vanilla_rrt/franka_reachable_morphit_100_goal_r025_20260805/`:
  trials, 100 saved path records, summary, plot, and validity audit
- `results/franka_flow_eb_rrt/franka_reachable_morphit_100_goal_r025_best249_20260805/`:
  trials, 100 saved path records, run config, summary, plot, and validity audit
- `results/franka_comparison/franka_reachable_morphit_100_goal_r025_best249_20260805/`:
  paired JSON comparison and plot

The large source datasets in `C:\Users\sodan\Downloads` were read in place and
were not copied or modified. Existing checkpoints, original benchmark data,
source datasets, old PyBullet results, and robot URDF/SRDF/mesh files were not
modified.

## Static-obstacle extension

The later static-obstacle evaluation is documented in
`scripts/FrankaObstacle/README.md`. It added a shared cuRobo/PyBullet cuboid
scene, 25 planner-independent obstacle-blocked problems, a combined
VanillaRRT/FlowEBRRT runner, cuboid-aware result auditing/comparison metadata,
and selectable PyBullet visualization. The completed results are in
`results/franka_obstacle/static_cuboid_25_20260805`.
Those completed planner results were generated before the runtime switch and
therefore use the historical 79-sphere model; they were not relabeled or
silently recomputed with 295 spheres.

## 295-sphere, radius-0.40 100-problem evaluation

The default normalized 7D joint-position goal radius in the VanillaRRT,
FlowEBRRT, and static-obstacle evaluation entry points is now 0.40. The active
Franka collision model remains the 295-sphere
`panda_morphit_d10_c1000_p1000.yml` fit.

The original fixed 100-problem set was audited against that active collision
model before rerunning the planners. Starts for all 100 problems and goals for
99 problems remained valid. The old goal and the last 16 witness states for
problem 40 collided under the denser model. To retain the benchmark as closely
as possible, `repair_franka_problem_set_for_active_collision.py` preserved 99
problems byte-for-byte at the array-row level and truncated only problem 40 to
its last valid witness state. The repaired set is
`data/benchmarks/franka_reachable_morphit295_ab_100_goal_r040.npz` (SHA-256
`5a98fcb5cfca279ce7f87c6b5642bb48ecf692ee3cae69bd09b3b22c8055fd5b`).
An independent audit passed all endpoint, dynamics, joint-limit,
self-collision, stored full-state witness-separation, and witness checks.

Both planners then ran sequentially on the same 100 problems, with a 30-second
per-problem budget and seed 20260721. Times below include successful problems
only.

| Planner | Successes | Success rate | Mean time | Median time |
| --- | ---: | ---: | ---: | ---: |
| VanillaRRT | 90/100 | 90% | 1.450 s | 0.159 s |
| FlowEBRRT (`best_inference.pt`, epoch 249) | 87/100 | 87% | 1.440 s | 0.0667 s |

Because the old problem set was reused, its selection distance includes both
joint positions and velocities, whereas planner goal completion uses joint
positions only. At radius 0.40, 7 reused starts (problems 2, 4, 8, 12, 24, 27,
and 39) therefore satisfy the planner goal immediately. On the remaining 93
nontrivial starts, VanillaRRT solved 83/93 (89.25%; successful-only mean 1.572
s, median 0.253 s) and FlowEBRRT solved 80/93 (86.02%; successful-only mean
1.566 s, median 0.0821 s). The primary table intentionally reports the full
100-problem evaluation requested.

The paired outcomes were 80 both succeeded, 10 VanillaRRT only, 7 FlowEBRRT
only, and 3 neither. Post-run audits found every saved path valid under the
active 295-sphere checker. Results and comparison artifacts are under:

- `results/franka_vanilla_rrt/franka_reachable_morphit295_100_goal_r040_20260805/`
- `results/franka_flow_eb_rrt/franka_reachable_morphit295_100_goal_r040_best249_20260805/`
- `results/franka_comparison/franka_reachable_morphit295_100_goal_r040_best249_20260805/`
