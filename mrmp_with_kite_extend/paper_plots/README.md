# Paper Plots

This folder contains the generated publication-plot artifacts. The code that
produces them lives in `paper_plotting_scripts/`.

## Layout

- `data/`: normalized CSV inputs used by the copied scripts.
- `figures/`: generated PNG images only.
- `csvs/`: generated CSV summaries for the plotted values.

The flatter figure layout keeps the plot type as the main folder and encodes
paradigm/system details in filenames, for example:

- `figures/computation_time/computation_time_kcbs_UCYCLE_all_envs_no_legend.png`
- `figures/success_rate/success_kcbs_all_systems.png`
- `figures/metrics/computation_time_all_paradigms.png`

The matching generated CSVs live under the same relative path in `csvs/`, for
example:

- `csvs/computation_time/computation_time_kcbs_UCYCLE_all_envs_no_legend.csv`
- `csvs/success_rate/success_kcbs_all_systems.csv`
- `csvs/metrics/computation_time_all_paradigms.csv`

## Current Scope

The plotting scripts read the current paper result set and write all generated
artifacts beneath this folder. dbCBS is normalized into:

- `data/dbcbs_normalized.csv`: dbCBS rows with explicit agent `radius`.

Generated CSVs include a `radius` column, meaning the robot/agent radius from
each run's `manifest.json` (`agent_builders[0].params.radius`). It is not the
goal radius from folder names like `_gr0.5_kd0.1`.

## Radius Naming

This result set uses UCYCLE agent radius 0.3 only. Generated plot and CSV
filenames no longer include a UCYCLE radius token. Generated CSV rows still
include a `radius` column for traceability.

## Aggregation

- Success-rate plots use `successes / total_trials` for each condition.
- Computation-time and total-path-time scaling plots use the arithmetic mean
  over successful trials only.
- `*_median*` heatmaps use median aggregation over condition-level ratios.
- Non-median ratio heatmaps use geometric mean aggregation over positive
  condition-level ratios.
- Paired heatmaps compute ratios only on shared successful seeds for non-dbCBS
  methods; dbCBS remains condition-level because it is imported from aggregate
  tables.

## Regenerating

From the repository root:

```bash
python paper_plotting_scripts/run_all_flat_plots.py
```
