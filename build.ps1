<#
    Build FilterAnnotationRefiner_Setup_v<VERSION>.exe

    Nuitka compiles the app, a smoke test proves the exe actually works, then
    Inno Setup packages it. Mirrors the FilterInspection pipeline; see
    PACKAGING.md for the reasoning behind each flag.

    Usage:   .\build.ps1
             .\build.ps1 -SkipLaunchTest      # machine without a GPU
             .\build.ps1 -SkipInstaller       # exe only, skip the slow Inno step
#>
[CmdletBinding()]
param(
    [switch]$SkipLaunchTest,
    [switch]$SkipInstaller
)

# Native tools write to stderr on success; Stop would abort the build on noise.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot

$Python  = ".\.venv-build\Scripts\python.exe"
$Iscc    = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
$BuildDir = "build"
$DistDir  = "dist"

function Fail($message) { Write-Host "BUILD FAILED: $message" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- preflight
if (-not (Test-Path $Python)) {
    Fail ".venv-build is missing. Create it per PACKAGING.md section 3.1. Never use conda."
}

# A conda-built exe dies at 'import ctypes' with 0xC0000409. Refuse outright.
$pythonHome = & $Python -c "import sys; print(sys.base_prefix)"
if ($pythonHome -match "conda|anaconda|miniconda|miniforge") {
    Fail "$Python resolves into a conda install ($pythonHome). Use a python.org venv."
}

$version = & $Python -c "import sys; sys.path.insert(0,'.'); from refiner.version import VERSION; print(VERSION)"
if (-not $version) { Fail "Could not read VERSION from refiner\version.py" }
Write-Host "Building version $version" -ForegroundColor Cyan

# sam2 is not on PyPI and the app adds it to sys.path at runtime, which Nuitka
# cannot follow. It must be importable at COMPILE time and named explicitly.
$sam2Repo = $env:SAM2_REPO
if (-not $sam2Repo) { Fail "Set SAM2_REPO to a SAM 2 checkout before building (PACKAGING.md 3.2)." }
if (-not (Test-Path (Join-Path $sam2Repo "sam2\__init__.py"))) {
    Fail "SAM2_REPO=$sam2Repo does not contain sam2\__init__.py"
}

$weights = Get-ChildItem -Path "weights" -Filter "sam2*.pt" -ErrorAction SilentlyContinue |
           Where-Object { $_.Length -gt 100MB } | Select-Object -First 1
if (-not $weights) {
    Fail "weights\ has no SAM 2 checkpoint over 100 MB. Copy sam2.1_hiera_large.pt there."
}
Write-Host "Checkpoint: $($weights.Name) ($([math]::Round($weights.Length/1GB,2)) GB)"

$gitStatus = & git status --porcelain
if ($gitStatus) { Write-Host "WARNING: working tree is dirty; this build includes uncommitted changes." -ForegroundColor Yellow }

# Remove-Item -Recurse dies on the torch tree under PowerShell 5.1.
if (Test-Path $BuildDir) { & cmd /c "rd /s /q `"$BuildDir`"" }
New-Item -ItemType Directory -Force -Path $BuildDir, $DistDir | Out-Null

# ---------------------------------------------------------------- compile
# PYTHONPATH carries sam2 for the compile; --include-package pulls it in, and
# --include-package-data brings its 13 hydra .yaml configs, which are loaded by
# filename at runtime and would otherwise be missing.
$env:PYTHONPATH = "$PSScriptRoot;$sam2Repo"

$nuitkaArgs = @(
    "-m", "nuitka",
    "--standalone",
    "--assume-yes-for-downloads",
    "--enable-plugin=pyside6",
    "--include-package=refiner",
    "--include-package=sam2",
    "--include-package-data=sam2",
    "--include-module=hydra",
    "--include-package=hydra",
    "--include-package-data=hydra",
    "--include-package=omegaconf",
    "--include-package=iopath",
    "--include-module=cv2",
    "--include-module=yaml",
    "--nofollow-import-to=pytest",
    "--nofollow-import-to=tkinter",
    "--windows-console-mode=disable",
    "--company-name=ITRI",
    "--product-name=Filter Annotation Refiner",
    "--file-version=$version.0",
    "--product-version=$version.0",
    "--output-dir=$BuildDir",
    "--output-filename=FilterAnnotationRefiner.exe",
    "app.py"
)
Write-Host "Compiling with Nuitka (expect 30-60 min on a cold cache)..." -ForegroundColor Cyan
& $Python @nuitkaArgs
$distApp = Join-Path $BuildDir "app.dist"
$exe = Join-Path $distApp "FilterAnnotationRefiner.exe"
if (-not (Test-Path $exe)) {
    Fail "$exe is missing after Nuitka reported success. Antivirus may have deleted it (PACKAGING.md 6)."
}

# The checkpoint ships beside the exe; refiner.paths.bundled_weights_dir() looks
# there using sys.argv[0], not __file__, so onefile extraction cannot confuse it.
New-Item -ItemType Directory -Force -Path (Join-Path $distApp "weights") | Out-Null
Copy-Item $weights.FullName (Join-Path $distApp "weights") -Force
Copy-Item "README.md", "LICENSE" $distApp -Force

# ---------------------------------------------------------------- smoke test
Write-Host "Smoke testing the compiled exe..." -ForegroundColor Cyan
& $exe --help 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "FilterAnnotationRefiner.exe --help exited $LASTEXITCODE" }

if (-not $SkipLaunchTest) {
    # Proves the GUI starts, Qt plugins resolved, and SAM 2 loads from the
    # bundled checkpoint. A window opens briefly; that is expected.
    $probe = Start-Process -FilePath $exe -ArgumentList "--self-test" -PassThru -Wait -NoNewWindow
    if ($probe.ExitCode -ne 0) {
        Fail "Self test exited $($probe.ExitCode). Run '$exe --self-test' by hand to see why."
    }
    Write-Host "Self test passed: GUI constructed, SAM 2 checkpoint loaded." -ForegroundColor Green
}

if ($SkipInstaller) { Write-Host "BUILD COMPLETE (exe only): $distApp" -ForegroundColor Green; exit 0 }

# ---------------------------------------------------------------- installer
if (-not (Test-Path $Iscc)) { Fail "Inno Setup 6 not found at $Iscc" }
Write-Host "Packaging installer (Inno Setup compresses several GB; ~10-15 min)..." -ForegroundColor Cyan
& $Iscc "/DMyAppVersion=$version" "/DMySourceDir=$distApp" "installer.iss"
if ($LASTEXITCODE -ne 0) { Fail "ISCC exited $LASTEXITCODE" }

$setup = Join-Path $DistDir "FilterAnnotationRefiner_Setup_v$version.exe"
if (-not (Test-Path $setup)) { Fail "$setup was not produced" }
$size = [math]::Round((Get-Item $setup).Length / 1GB, 2)
Write-Host ""
Write-Host "BUILD COMPLETE" -ForegroundColor Green
Write-Host "  $setup  ($size GB)" -ForegroundColor Green
