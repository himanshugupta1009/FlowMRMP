# Paper plotting scripts

This folder contains the code for normalizing experiment results and producing
publication plots. Generated artifacts are kept separately:

- `paper_plots/` for the canonical paper results.
- `brad_plots/` for Brad's 2026-07-15 result bundle.

Run from the repository root:

```bash
python paper_plotting_scripts/run_all_flat_plots.py
python paper_plotting_scripts/run_brad_plots.py
```

Individual `plot_*.py` scripts also write to `paper_plots/` by default.
