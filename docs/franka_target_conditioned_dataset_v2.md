# Franka target-conditioned dataset v2

## Purpose

V2 fixes the main v1 limitation. V1 could only reorder the 32 edges already
stored for each source query. V2 reconstructs new 128-candidate neighborhoods,
fully re-integrates and validates them from each exact query state with the
active 295-sphere MorphIt/cuRobo model, and retains every valid candidate from
the accepted tested pools.

## Source and output

Immutable source:

`franka_edge_bundle_200k_pool128_k32_n350000_max50_fullvalid.h5`

V2 companion:

`franka_edge_bundle_200k_pool128_k32_n350000_max50_targetcond_v2.h5`

The companion references the source by basename and SHA-256. It does not
duplicate the 200,000 trajectories or 24.3 million raw-edge records. Both files
must remain in the same directory when transferred to another machine.

## Current collision-model filtering

The legacy source was generated under an earlier collision baseline. Before v2
pool generation, all source query configurations are rechecked with the active
295-sphere MorphIt/cuRobo model. Initial configurations rejected by the current
model are removed and their original indices remain recoverable from the
source. Accepted v2 rows store `base_source_index`.

A state that passes the initial collision check can still be unable to supply
32 complete valid control sequences. V2 tests deterministic pool128 draws at
the legacy radius, expands sparse neighborhoods through radius multipliers
`1.25`, `1.5`, and `2.0`, and keeps every dynamics and collision requirement
unchanged. If fewer than 32 valid controls exist after that search, the source
index is stored in `rejected_source_indices` and the state is skipped. Invalid
or duplicate controls are never used as padding.

## Targets and selected bundles

Each accepted base state produces two 29D target-conditioned rows:

1. a position-only goal drawn from the fixed 100-problem benchmark goal
   distribution; and
2. a reachable full-state target selected from a validated pool endpoint.

Each final K=32 bundle contains:

- eight candidates with highest actual normalized target progress; and
- 24 candidates selected for endpoint/control-duration diversity by FPS.

The flat variable-length pool is also retained. This allows later selection
experiments without repeating dynamics or CuRobo collision validation.

## Hard examples

The training split appends 104 sampled intermediate states from 13 failed
FlowEBRRT trials in the fixed 100-problem run. Each hard state is paired with
its original benchmark goal, current-model collision checked, and processed
through the same pool128 validation pipeline. Validation contains no hard
states and stays independent.

## Resumability

Generation records both:

- `input_cursor`: how many filtered source/hard states were scanned; and
- `completed_rows`: how many valid base rows were committed.

Pool arrays are committed before the cursor advances and flushed after every
checkpoint batch. On resume, an uncommitted flat-pool tail is truncated to the
last committed offset. After all inputs are scanned, fixed-capacity datasets
are compacted to the exact accepted row count before the completion flag is
set.

## Full generation command

```powershell
python scripts/create_franka_target_conditioned_edge_dataset_v2.py `
  --output C:\Users\sodan\Downloads\franka_edge_bundle_200k_pool128_k32_n350000_max50_targetcond_v2.h5 `
  --checkpoint-every 32 `
  --pool-retries 20
```

Running the same command resumes an incomplete output automatically.

## Required audit

```powershell
python scripts/audit_franka_target_conditioned_edge_dataset_v2.py `
  C:\Users\sodan\Downloads\franka_edge_bundle_200k_pool128_k32_n350000_max50_targetcond_v2.h5
```

The completed file must not be used for training unless this audit reports
`"passed": true`. The training loader also rejects an incomplete v2 file.

## Training

The updated `train_franka_edge_flow_matching.py` recognizes v2 directly. After
the full audit passes, place the v2 companion and immutable source together in
`dataset/franka_eb_dataset/` and point `--dataset` at the v2 companion. On the
512 GB RAM training machine, keep `--cache-encoded-dataset` enabled.
