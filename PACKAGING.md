# Packaging Filter Annotation Refiner for Windows

Design and background. The operating procedure is in
`.claude/skills/package-release/SKILL.md`.

## 1. What the installer contains

| Component | Size | Why it must be bundled |
|---|---|---|
| CUDA torch + torchvision | ~1.7 GB | Vendor machines have no Python toolchain |
| PySide6 (Qt) | ~0.7 GB | GUI |
| OpenCV, numpy, Pillow, PyYAML | ~0.13 GB | Geometry and image IO |
| SAM 2 package + 13 hydra configs | ~2 MB | Loaded by filename at runtime |
| `sam2.1_hiera_large.pt` | 857 MB | A conversion cannot run without it |

Roughly **3.4 GB uncompressed, ~1.5-2 GB compressed**, comparable to the
FilterInspection installer.

## 2. Versioning

`refiner/version.py` holds the single `VERSION`. It drives the window title, the
installer filename and the installer's `AppVersion`. `tests/test_cli.py` asserts
it matches `pyproject.toml`, because those two drifted apart once already
(1.1.0 vs 1.0.0).

- **patch** - a fix or a threshold change
- **minor** - a new compatible feature
- **major** - an output-layout change, or anything that invalidates a dataset
  produced by the previous version

## 3. Build environment

### 3.1 The build venv

```powershell
py -3.12 -m venv .venv-build
.\.venv-build\Scripts\python.exe -m pip install -U pip wheel
.\.venv-build\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv-build\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-build\Scripts\python.exe -m pip install nuitka ordered-set zstandard
.\.venv-build\Scripts\python.exe -m pip install hydra-core iopath
```

**Never build from conda.** A conda-built exe dies at `import ctypes` with
`0xC0000409`. `build.ps1` reads `sys.base_prefix` and refuses if it matches
conda/anaconda/miniforge.

### 3.2 SAM 2

SAM 2 is not on PyPI, and `refiner/backend/sam2_backend.py` adds a checkout to
`sys.path` **at runtime** — which Nuitka cannot follow. So it must be importable
at *compile* time and named explicitly:

```powershell
$env:SAM2_REPO = "C:\sam2"
```

`build.ps1` puts that on `PYTHONPATH` for the Nuitka call and passes
`--include-package=sam2 --include-package-data=sam2`. The `-package-data` part
is not optional: SAM 2 loads one of 13 `.yaml` hydra configs by filename, and
without it the build compiles cleanly and then fails at first inference.

When frozen, `_ensure_sam2_importable()` skips the disk search entirely and
raises a message naming the missing build flag — searching a vendor machine for
a checkout it will never have only obscures the real cause.

### 3.3 Weights

Put the checkpoint in `weights\` before building. `build.ps1` refuses if nothing
there exceeds 100 MB, and copies it to `app.dist\weights\`.

At runtime `refiner/paths.py:bundled_weights_dir()` resolves it from
**`sys.argv[0]`**, not `__file__`. In a onefile build `__file__` is a temporary
extraction folder, so anything that must be found in the *install* folder has to
use `sys.argv[0]`.

`weights/` is gitignored. Never commit an 857 MB checkpoint.

### 3.4 Licensing before shipping weights

SAM 2.1 weights come from Meta under their own terms, and the local checkout is
a third-party fork. **Confirm redistribution is permitted before handing an
installer containing them to a vendor.** If it is not, ship without `weights\`
and have the operator supply a checkpoint — discovery already falls back to
`$SAM2_CHECKPOINT` and common locations.

## 4. Build

```powershell
.\build.ps1
.\build.ps1 -SkipLaunchTest    # no GPU on the build machine
.\build.ps1 -SkipInstaller     # exe only
```

Log outside the repo; a file open in an editor is locked and the pipe fails
before the build starts:

```powershell
.\build.ps1 *>&1 | Out-File "$env:TEMP\far_build.log" -Encoding utf8
```

## 5. The smoke test

Before the slow Inno step, `build.ps1` runs `FilterAnnotationRefiner.exe
--self-test`, which:

1. constructs the main window and checks it became visible — catches missing Qt plugins
2. loads the segmentation backend — catches sam2 not compiled in, missing hydra configs
3. runs **one real inference** on a probe box — catches a checkpoint that was not shipped

Each failure prints a specific cause. A build that passes `--help` but fails
`--self-test` is not shippable; diagnose it rather than passing `-SkipLaunchTest`.

## 6. When it fails

| Symptom | Cause / fix |
|---|---|
| exe exits `-1073740791` (0xC0000409) | Built from conda. Use `.venv-build`. |
| `No module named 'sam2'` from the exe | `--include-package=sam2` missing, or `SAM2_REPO` not on `PYTHONPATH` for the Nuitka call. Nuitka cannot follow the runtime `sys.path.insert`. |
| Loads, then fails at first inference with a hydra/config error | `--include-package-data=sam2` missing; the 13 `.yaml` configs were not bundled. |
| `No SAM 2 checkpoint found` on a vendor machine | `weights\` not copied into `app.dist`, or code used `__file__` instead of `sys.argv[0]`. |
| Blank window, no error | Qt plugins missing: `--enable-plugin=pyside6` dropped. |
| exe missing after Nuitka said success | Antivirus deleted it. Rebuild; get `build\` excluded by IT if it recurs. |
| Nuitka prompts for a download and dies | `--assume-yes-for-downloads` removed from build.ps1. |
| Build log stops with no error | `Remove-Item -Recurse` on the torch tree kills PowerShell 5.1. Script uses `rd /s /q`; keep it. |
| `Out-File : user-mapped section open` | Log file open elsewhere. Log to `$env:TEMP`. |

Diagnose the root cause before editing `build.ps1`. Do not paper over a failing
smoke test.

## 7. Acceptance checklist on a clean machine

This procedure cannot verify GPU throughput or vendor hardware. Before handing
the installer over, on a machine that has never had the app:

1. Install; confirm the NVIDIA-driver warning appears only if no driver is present.
2. Launch from the Start menu. Title shows the expected version.
3. Switch to 繁體中文 and back.
4. Load a small detection dataset; the Input annotations preview draws boxes.
5. Set **Fast**, convert ~20 images. Expect roughly 0.3 s/image on a GPU.
6. Confirm the output has `dataset\` (images, labels, data.yaml) and `_debug_and_logs\`.
7. Train one epoch on `dataset\data.yaml` to prove the labels load.
8. Uninstall; confirm the install folder is removed.

Do not commit `build\`, `dist\`, `.venv-build\` or `weights\` — all gitignored.
