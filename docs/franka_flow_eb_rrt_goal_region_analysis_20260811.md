# Franka FlowEBRRT goal-region and solution-quality analysis

Date: 2026-08-11

## Scope

This analysis uses the existing audited 100-problem, empty-world Franka A-B
benchmark with a normalized 7D joint-position goal radius of 0.40, 295
MorphIt/cuRobo robot spheres, a 30 second per-problem budget, and the epoch-249
`best_inference.pt` checkpoint. No benchmark paths were regenerated for the
100-problem quality comparison.

The focused edge audits reran two single-agent problems with the same planner
configuration while retaining copies of the generated controls for offline
classification:

- Problem 61: representative Vanilla-only success / Flow failure.
- Problem 0: representative Flow success.

## Solution-quality definition

`path_motion_time_seconds` is physical trajectory duration from the start state
through the first integration waypoint that satisfies the goal test. With the
current 20 ms integration interval, it is `(number_of_dense_states - 1) * 0.02`.
It excludes planner wall time, model inference, collision checking, and file I/O.
Only successful paths have a finish time; failures remain `NaN`/`null`.

The benchmark code now recomputes this value from the dense path, checks it
against the tree's accumulated edge durations, stores it per path/trial, and
reports aggregate successful-path statistics and ratios to the known feasible
witness duration.

## 100-problem results

| Metric | VanillaRRT | FlowEBRRT |
|---|---:|---:|
| Successes | 90/100 | 87/100 |
| Easy | 41/41 | 41/41 |
| Medium | 38/39 | 36/39 |
| Hard | 11/20 | 10/20 |
| Successful path time, mean | 0.601 s | 0.884 s |
| Successful path time, median | 0.330 s | 0.340 s |
| Tree nodes, all trials | 6,555 | 53,948 |
| Checked waypoints, all trials | 1,305,416 | 1,391,434 |

The cleanest quality comparison is the 80 problems solved by both planners:

| Both-success metric | VanillaRRT | FlowEBRRT |
|---|---:|---:|
| Mean path time | 0.4848 s | 0.5970 s |
| Median path time | 0.3200 s | 0.3100 s |
| 95th percentile | 1.461 s | 1.920 s |

Flow is tied at the median but has a worse tail. Its mean is 0.1123 s (23.2%)
longer on the paired both-success set. This is not a universal loss: Flow solves
seven problems Vanilla misses, including several long hard-case trajectories.

Outcome IDs:

- Vanilla only: 61, 64, 69, 80, 82, 84, 88, 91, 97, 99.
- Flow only: 68, 85, 86, 89, 90, 92, 96.
- Neither: 81, 83, 98.

## What the 32 edges actually do

At a selected node, FlowEBRRT generates and caches 32 candidates. They are not
all propagated immediately. For the current sampled RRT target, unused
candidates are ranked by their learned terminal-state prediction. The planner
tries them in that order and stops as soon as the first valid candidate adds a
tree node or reaches the goal. Attempted candidates are permanently masked.

If that node is selected again, only still-unmasked candidates participate. If
all 32 are masked, the Flow candidate loop has zero entries and the planner goes
directly to its one random-control fallback. It never regenerates a fresh Flow
bundle for that node.

Across the saved 100-problem run, Flow attempted only 0.965 learned edges per
RRT iteration on average. The nominal K=32 is therefore a generated local
choice set, not an effective branching factor of 32.

## Focused failure: problem 61

Vanilla solves problem 61 in 1.43 s of planning with a 1.10 s physical path.
Flow times out after 30 s; its closest tree node is 0.5071 from the goal, outside
the 0.40 radius.

Within a diagnostic radius of 0.80:

- 61 near-goal tree nodes existed.
- 59 had generated bundles, partly because of prefetch.
- Only 32 were ever selected for expansion.
- Only one node consumed all 32 candidates.
- The closest node was selected four times and had tried only 4/32 candidates;
  it still had 28 unused candidates at timeout.
- Replaying all 32 candidates at that closest node found no candidate that even
  improved on its 0.5071 parent distance.

Across the 50 closest generated bundles (1,600 exact candidates):

- 0 learned terminal predictions were inside the goal radius.
- 0 physically propagated paths crossed the goal radius.
- 1,448 were valid but did not reach the goal.
- 145 violated joint-position or joint-velocity limits.
- 7 self-collided.
- The best waypoint among all 1,600 candidates was still 0.4835 from the goal.

The single exhausted node was at distance 0.6329. All 32 of its candidates
violated state limits. It was selected twice; each selection ended in a random
fallback because no Flow candidates remained.

This rejects two tempting explanations for problem 61: collision checking is
not the dominant loss (7/1,600 candidates), and simply forcing all 32 candidates
to be tried would not produce a goal-reaching local edge in the audited region.

The learned endpoint predictions were also reasonably accurate on this run:
median absolute endpoint-distance error was 0.00346 and the 95th percentile was
0.0118. The principal issue is missing goal-directed support in the generated
candidate set, not gross endpoint-prediction error.

## Successful contrast: problem 0

Problem 0 succeeds with a 0.22 s physical path. At its selected near-goal node,
all 32 retained candidates physically crossed the goal region, and all 32
predictions were inside it. The planner tried only the first candidate and
stopped. This confirms that prefix goal detection works and that the model can
provide excellent local support in states that match a well-covered motion
mode.

## Why FlowEBRRT is not outperforming VanillaRRT

1. **The learned edge distribution is state-conditioned, not target- or
   goal-conditioned.** The network receives the current 14D Franka state and
   generates a generic local set. The sampled RRT target is used only after
   generation to rank those 32 candidates. Vanilla instead samples 16
   control-duration pairs and chooses using their physically propagated
   endpoints. Near a particular goal direction, Flow can have no candidate
   with the required braking/turning behavior even when the model accurately
   predicts its candidates.

2. **Generated does not mean expanded.** Most valid Flow extensions consume one
   candidate and return. A near-goal node must win nearest-neighbor selection
   repeatedly before its remaining candidates are tried. Problem 61's closest
   node received only four selections in roughly 2,852 iterations.

3. **The training bundles encode local demonstration support.** Training
   selects 32 diverse, valid trajectory suffixes from a nearby pool of 128 and
   uses farthest-point sampling over duration and outcomes. This preserves
   demonstrated local modes but does not guarantee coverage of every arbitrary
   query-to-goal direction. The 100 A-B endpoints are in-distribution, but an
   RRT tree reaches intermediate states that need not lie on the witness
   trajectory's exact state/direction sequence.

4. **Flow has a local-capture weakness but a useful long-motion tail.** Both
   planners solve every easy case. Flow loses two cases in the 0.4-0.8 initial
   distance band and trails 5-8 in the 1.2-1.6 band, yet wins 4-3 in the
   greater-than-1.6 band and has seven unique successes. The learned primitives
   sometimes supply useful long-horizon structure, but do not close the last
   local gap reliably.

5. **Inference consumes budget without fixing support.** Flow generated 51,161
   bundles in 3,668 batched model calls. Bundle generation used 223.8 s, 43.4%
   of summed Flow planner wall time. Prefetch averaged 13.95 bundles per call.
   Faster inference would buy more iterations, but problem 61 already built
   thousands of nodes and 1,600 audited near-goal candidates still contained no
   local connector.

6. **Path quality has a heavier tail.** On both-success problems the medians are
   essentially tied, but Flow's 95th-percentile path is 1.920 s versus 1.461 s.
   This is consistent with generic primitives chaining through indirect motion
   modes rather than directly steering to the task goal.

## Recommended order of experiments

1. Condition edge generation on normalized `target - current` (or explicitly
   on the goal when goal-biased sampling is active), not just the current state.
2. Add a goal-region local connector: when position-goal distance is below a
   threshold, generate targeted braking/acceleration controls or mix in a much
   larger target-scored random-control batch.
3. For goal-biased samples, rank by 7D position-goal distance or a goal-aware
   score. The current learned ranking operates in full normalized 14D state,
   while success ignores goal velocity.
4. Measure candidate-set recall during training and validation: fraction of
   query/target pairs for which any of K propagated edges reaches or makes a
   required amount of progress toward the target.
5. Retrain with hard intermediate tree states and an explicit goal-progress or
   local-connectivity objective. The current dataset validates physical edges
   but does not label whether a set can connect an arbitrary downstream target.
6. Only then test K=64/128. More candidates help only if they add directional
   support; problem 61's existing 32 candidates were already mostly valid but
   all missed the goal.
7. Reduce or adapt prefetch after local reliability is fixed. This is a runtime
   optimization, not the primary success-rate fix.

## Literature connection

- Original RRT work emphasizes broad state-space exploration and suitability
  for differential constraints: Steven M. LaValle,
  [*Rapidly-Exploring Random Trees: A New Tool for Path Planning*](https://lavalle.pl/papers/Lav98c.pdf)
  (1998).
- Ichter, Harrison, and Pavone mix learned and uniform sampling so learned
  support can accelerate search without eliminating coverage when the learned
  distribution misses a needed region:
  [*Learning Sampling Distributions for Robot Motion Planning*](https://arxiv.org/abs/1709.05448)
  (ICRA 2018).
- Qureshi et al. likewise recover worst-case guarantees by combining a neural
  planner with a classical sampling-based fallback:
  [*Motion Planning Networks: Bridging the Gap Between Learning-based and Classical Motion Planners*](https://arxiv.org/abs/1907.06013).
- The State Supervised Steering Function formulation learns a policy of both
  current state and desired target state, directly matching the missing
  conditioning identified here: Chiang et al.,
  [*State Supervised Steering Function for Sampling-based Kinodynamic Planning*](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p35.pdf)
  (AAMAS 2022).

## Artifacts

- `scripts/FrankaTesting/franka_solution_quality.py`
- `scripts/FrankaTesting/analyze_franka_flow_goal_region.py`
- `results/franka_flow_analysis/quality_comparison_20260811/`
- `results/franka_flow_analysis/goal_region_problem_061_20260811/`
- `results/franka_flow_analysis/goal_region_problem_000_success_20260811/`
