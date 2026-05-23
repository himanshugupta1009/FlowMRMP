# FlowMRMP Integration Strategy

## Goal

FlowMRMP should become the working repo for MRMP experiments that use learned
diffusion or flow-matching guidance. The MRMP planners should remain the source
of truth for planning, collision checking, multi-agent coordination, and agent
dynamics. The DiTree code should be preserved as the source of training and
inference components for learned action proposals.

## Recommended Layout

```text
FlowMRMP/
  ditree/                       # current DiTree-derived code moved intact
    train_diffusion_policy.py
    train_manager.py
    rollout_manager.py
    run_scenarios.py            # keep for reference/benchmarking, not primary MRMP entrypoint
    car_env.py
    maze_datasets.py
    local_map_encoder.py
    log_to_tensorboard.py
    common/
    policies/
    model/
    cfgs/
    maps/
    experiments/
    metadata/

  mrmp_with_kite_extend/        # git subtree from himanshugupta1009/mrmp_with_kite_extend
    src/
    pipeline_code/
    Tests/
    trial_scripts/
    animators/
    visualizations/
    README.md
    requirements.txt

  src/                          # new FlowMRMP integration code
    flow_mrmp/
      __init__.py
      guidance/
        ditree_policy.py        # loads model/checkpoint and exposes action proposals
        local_map_adapter.py    # converts MRMP env/agent state into DiTree local-map tensors
      planners/
        guided_rrt.py           # MRMP RRT subclass or strategy using learned proposals

  scripts/
    train_ditree_policy.py      # thin wrapper around ditree training
    run_guided_mrmp.py          # experiment entrypoint for MRMP + learned guidance

  artifacts/                    # local only: checkpoints, datasets, generated plots
```

## Why This Shape

The MRMP repository already has the planner abstractions we want to keep:

- `src/rrt.py` chooses controls in `_select_best_extension_candidate`.
- `src/edge_bundle_rrt.py` shows the right pattern for guided extension: subclass
  `RRT`, override `extend_tree`, try guided candidates first, then fall back.
- agent dynamics live on the agent objects, for example
  `UniCycle.get_next_state` and `UniCycle.get_random_action`.

The DiTree code already has the reusable learned-policy pieces:

- `train_diffusion_policy.py` owns training.
- `policies/fm_policy.py::DiffusionSampler` owns model inference and returns
  candidate action or observation sequences.
- `common/map_utils.py` and related files own local map features.

So the useful boundary is not "replace MRMP RRT with DiTree RRT." The useful
boundary is: MRMP asks a learned guidance object for candidate controls, then
MRMP remains responsible for propagation, validity checks, goal checks, CBS
constraints, and final path extraction.

## First Integration Target

Start with single-agent `UniCycle` RRT before touching K-CBS or cRRT.

1. Move current DiTree files into `ditree/`.
2. Add `mrmp_with_kite_extend/` as a git subtree from
   `git@github.com:himanshugupta1009/mrmp_with_kite_extend.git`, excluding
   virtualenvs and generated outputs.
3. Add `flow_mrmp.guidance.DiTreeActionGuide` with a small API:

   ```python
   actions = guide.propose_actions(
       state=parent_node.state,
       goal=self.goal,
       env=self.env,
       agent=self.agent,
       previous_actions=previous_actions,
       num_candidates=self.num_extension_trials,
   )
   ```

4. Add `GuidedRRT(RRT)` that tries learned actions first and uses
   `agent.get_random_action` as fallback.
5. Once single-agent behavior is stable, plug `GuidedRRT` into the existing MRMP
   K-CBS/pRRT/cRRT test harnesses.

## Data And Artifact Policy

Do not commit these by default:

- virtualenvs: `venv/`, `.venv/`, `venv_mapf/`
- DiTree checkpoints and datasets: `checkpoints/`, `data/`
- generated results: `tree_plot/`, `png_exports/`, `benchmark_results/`,
  `path_outputs/`, `test_results/`

Large reusable assets such as edge bundles and motion primitives should be a
deliberate choice. If they are needed for repeatable experiments, keep small
default examples in git and place full-size files in `artifacts/` or use Git LFS.

## Practical Next Step

Do the restructure in two commits:

1. Pure file move/import commit: no behavior changes.
2. Adapter commit: add learned action guidance and a single guided-RRT test.

That keeps regressions easy to diagnose.
