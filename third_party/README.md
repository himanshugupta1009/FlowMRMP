# Third-party runtime dependency

The cuRobo/MorphIt work uses NVlabs cuRobo cloned at:

- repository: `https://github.com/NVlabs/curobo.git`
- commit: `8e734f3ced1df898990bcd92de40abce475907db`
- reported version: `0.8.0.post1.dev42`

The checkout and its isolated Windows virtual environment are intentionally
ignored by the parent repository. Generated Franka configuration and benchmark
artifacts remain in the parent repository.

Recreate the pinned environment from the repository root with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_franka_curobo.ps1
```

The setup uses Python 3.12, PyTorch 2.8.0 with CUDA 12.6, the `cu12` cuRobo
extras, `viser==1.0.30` (the older resolver choice depended on a Windows-missing
`liblzfse` wheel), and `tifffile==2025.5.10` to remain compatible with the
project's pinned NumPy 2.0.2.
