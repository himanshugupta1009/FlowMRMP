# Franka target-conditioned training package v1

## Outcome

This package prepares the existing Franka edge-bundle data and training code
for two new planner inputs:

1. the requested target relative to the current Franka state; and
2. a goal-mode flag that distinguishes a full 14D state target from the
   position-only goal test used by the planner.

The original trajectory HDF5 is not modified. A compact companion HDF5 stores
the new conditions and target-dependent edge order while continuing to use the
original file for controls, outcomes, and trajectories.

Full training is intentionally left for the GPU training machine. A one-epoch
debug run on this machine verifies the complete loader, model, loss, metrics,
and checkpoint path; it is not a quality result.

## Files to transfer

Keep these two HDF5 files in the same directory on the training machine:

| File | Size | SHA-256 |
|---|---:|---|
| `franka_edge_bundle_200k_pool128_k32_n350000_max50_fullvalid.h5` | 7,544,249,212 bytes | `7444be109233803a663897d9a1f9e7c0db82568bf4f4b3e809d7faa77f8636d7` |
| `franka_edge_bundle_200k_pool128_k32_n350000_max50_targetcond_v1.h5` | 139,676,889 bytes | `f935ff3a793638b50af631fff06428c0d496140dc8e2d9387960f23649368823` |

Also transfer the repository at the same revision, including:

- `scripts/create_franka_target_conditioned_edge_dataset.py`
- `scripts/audit_franka_target_conditioned_edge_dataset.py`
- `scripts/train_franka_edge_flow_matching.py`
- `Tests/test_franka_target_conditioning.py`

The companion records both the source basename and its SHA-256. The loader
first looks beside the companion, which makes the pair portable even though
the generation machine's absolute source path is also retained for provenance.

These are the only required files that must be shared separately from GitHub.
They are ignored by Git because they are large generated data artifacts. The
small `.audit.json` file may also be shared, but it is optional because the
receiving machine can regenerate it. The exact file list and checksums are also
stored in `docs/franka_target_conditioned_transfer_manifest_v1.json`.

After cloning the repository, place both required files here:

```text
FrankaFM/
  dataset/
    franka_eb_dataset/
      franka_edge_bundle_200k_pool128_k32_n350000_max50_fullvalid.h5
      franka_edge_bundle_200k_pool128_k32_n350000_max50_targetcond_v1.h5
```

## Dataset contract

The companion format is `franka_target_conditioned_edge_bundle`, version 1.
It contains:

- 665,000 training rows: 332,500 full-state and 332,500 goal-mode;
- 35,000 validation rows: 17,500 full-state and 17,500 goal-mode;
- 32 physically valid source edges per row;
- 8 target-directed slots followed by 24 source-order slots.

The 29D condition layout is:

| Slice | Meaning |
|---|---|
| `0:14` | normalized current `[q, dq]` |
| `14:28` | normalized relative requested target `[delta_q, delta_dq]` |
| `28` | goal mode (`0` full-state, `1` position-only) |

For configuration, the relative normalization is `2 * delta_q / q_range`.
For velocity it is `delta_dq / dq_max`. Goal-mode rows set relative target
velocity to zero, and target distance/recall ignore velocity for those rows.

Each target is a reachable terminal state selected from the row's integrated,
valid source-edge outcomes. Slots `0:8` are the eight outcomes closest to that
target under the applicable full-state or position-only normalized metric.

Important v1 limitation: this companion creates target-aware views of the
existing 32 candidates. It does not rescan or regenerate from the original
128-candidate pool, and it does not add failed-search-tree or obstacle-specific
examples. A later dataset generation pass should do those things after this
conditioning interface is proven by training and planner evaluation.

## Training changes

The training script now:

- detects both the original 14D format and the new 29D companion format;
- resolves and verifies the immutable source dataset;
- applies repeated base-row indices and target-specific edge permutations;
- builds the model with the dataset's actual condition width;
- preserves the target-directed slot layout for the Transformer slot tokens;
- adds a differentiable best-of-32 target-progress loss to the original flow
  matching loss;
- records base flow loss, target progress, and target recall in CSV metrics and
  checkpoints;
- embeds the target-conditioning contract and source hash in checkpoints.

The total loss is:

`base_flow_loss + target_progress_weight * mean(best_target_distance)`

The default target-progress weight is `0.25`. Target recall uses a normalized
radius of `0.40`; this is a training diagnostic, separate from the planner's
raw joint-space goal radius.

## Verify before training

From the repository root:

```powershell
python scripts/audit_franka_target_conditioned_edge_dataset.py `
  D:\data\franka_edge_bundle_200k_pool128_k32_n350000_max50_targetcond_v1.h5

python -m unittest Tests.test_franka_target_conditioning
```

The committed audit for the generated full dataset passed with no errors, no
chunk failures, and zero maximum distance between every stored target and its
recorded anchor outcome.

## Full GPU training command

The training script now defaults to the target-conditioned dataset at the
location above. From the repository root on Windows PowerShell:

```powershell
python scripts/train_franka_edge_flow_matching.py `
  --output-dir trained_models\franka_target_conditioned_v1 `
  --description target_conditioned_v1 `
  --device cuda `
  --epochs 60 `
  --batch-size 128 `
  --target-progress-weight 0.25 `
  --target-recall-radius 0.40 `
  --no-cache-encoded-dataset
```

The equivalent Linux command is:

```bash
python scripts/train_franka_edge_flow_matching.py \
  --output-dir trained_models/franka_target_conditioned_v1 \
  --description target_conditioned_v1 \
  --device cuda \
  --epochs 60 \
  --batch-size 128 \
  --target-progress-weight 0.25 \
  --target-recall-radius 0.40 \
  --no-cache-encoded-dataset
```

To resume an interrupted run, pass its full `last.pt` checkpoint:

```powershell
python scripts/train_franka_edge_flow_matching.py `
  --resume trained_models\franka_target_conditioned_v1\RUN_NAME\checkpoints\last.pt `
  --device cuda `
  --no-cache-encoded-dataset
```

Use `--cache-encoded-dataset` only on a machine with enough host RAM for the
entire expanded 700,000-by-32 encoded set. The non-cached command is the safer
starting point. Batch size can be raised after checking GPU memory use.

## Smoke verification result

The local debug run used 128 training rows, 32 validation rows, a reduced CPU
model, and one epoch. It completed and wrote both full and inference-only
checkpoints with `cond_dim=29` and the target metadata. Its validation loss was
`0.7135015`. Its zero one-epoch validation recall is expected from this tiny
debug run and is not evidence about the full model.

The production planner/inference adapter has deliberately not been changed in
this step. After the remote model is trained, the next integration step is to
construct the same 29D condition at every FlowEBRRT expansion, load the new
checkpoint, and rerun the fixed 100-problem comparison.
