[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$CuRoboCommit = "8e734f3ced1df898990bcd92de40abce475907db"
$ThirdPartyRoot = Join-Path $RepositoryRoot "third_party"
$CuRoboRoot = Join-Path $ThirdPartyRoot "curobo"
$EnvironmentRoot = Join-Path $CuRoboRoot ".venv-curobo-windows"
$EnvironmentPython = Join-Path $EnvironmentRoot "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $ThirdPartyRoot | Out-Null
if (-not (Test-Path (Join-Path $CuRoboRoot ".git"))) {
    git clone https://github.com/NVlabs/curobo.git $CuRoboRoot
}
git -C $CuRoboRoot fetch origin $CuRoboCommit
git -C $CuRoboRoot checkout --detach $CuRoboCommit

if (-not (Test-Path $EnvironmentPython)) {
    & $Python -m venv $EnvironmentRoot
}

& $EnvironmentPython -m pip install --upgrade pip setuptools wheel ninja
& $EnvironmentPython -m pip install `
    torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
& $EnvironmentPython -m pip install -e "$CuRoboRoot[cu12]"
& $EnvironmentPython -m pip install -r (Join-Path $RepositoryRoot "requirements-franka.txt")
& $EnvironmentPython -m pip install `
    viser==1.0.30 tifffile==2025.5.10 pybullet==3.2.7 pandas pytest
& $EnvironmentPython -m pip check

& $EnvironmentPython -c `
    "import torch, curobo; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('curobo', curobo.__version__)"

Write-Host "cuRobo environment ready: $EnvironmentPython"
