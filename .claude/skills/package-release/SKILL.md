---
name: package-release
description: Build the Filter Annotation Refiner Windows installer (FilterAnnotationRefiner_Setup_vX.Y.Z.exe) after app code changes. Use when the user asks to package, build, release, or make a new installer / 打包 / 出新版 / 做安裝檔.
---

# Package a Filter Annotation Refiner release

Design and background live in `PACKAGING.md`. This is the operating procedure.
**It runs on Windows only** — Nuitka is not a cross-compiler and Inno Setup is
Windows-only. On Linux, stop and tell the user.

## 1. Preflight (stop and tell the user if any fails)

- `.venv-build\Scripts\python.exe` exists. If not, build it with PACKAGING.md 3.1.
  **Never build from conda** — the exe crashes on `import ctypes`. `build.ps1`
  refuses conda anyway.
- `C:\Program Files (x86)\Inno Setup 6\ISCC.exe` exists.
- `$env:SAM2_REPO` points at a checkout containing `sam2\__init__.py`. SAM 2 is
  not on PyPI and the app adds it to `sys.path` at runtime, which Nuitka cannot
  follow (PACKAGING.md 3.2).
- `weights\` holds a SAM 2 checkpoint over 100 MB.
- Redistribution of those weights is permitted (PACKAGING.md 3.4). If unclear,
  **stop and ask the user** — do not ship third-party weights on an assumption.
- `git status` is clean, or the user confirms the uncommitted changes belong in
  this release.

## 2. Version

Ask which part to bump unless the user said so, per PACKAGING.md 2 (patch =
fix/tuning, minor = new compatible feature, major = output-layout change or
anything that invalidates datasets from the previous version). Edit `VERSION` in
`refiner\version.py`, and `version` in `pyproject.toml` to match — a test
asserts they agree. That single value drives the window title, the installer
name and the installer's AppVersion.

## 3. Build

From the repo root, in PowerShell, in the background. Expect 30-60 min on a cold
Nuitka cache, of which ~10-15 min is Inno Setup compressing several GB. Don't
poll — wait for completion.

```powershell
.\build.ps1
```

Log OUTSIDE the repo; a file open in an editor is locked and the pipe fails
before the build starts:

```powershell
.\build.ps1 *>&1 | Out-File "$env:TEMP\far_build.log" -Encoding utf8
```

It compiles the exe, copies the checkpoint beside it, then smoke-tests before
the slow Inno step: `--help`, and `--self-test`, which constructs the main
window, loads the segmentation backend and runs one real inference. A window
opens briefly — expected. On a machine without a GPU use `-SkipLaunchTest`,
but then the model path is unverified and must be checked on the target.

Output: `dist\FilterAnnotationRefiner_Setup_v<VERSION>.exe` (~1.5-2 GB).

A successful run prints, in order: Nuitka's success line, `self-test: main
window constructed`, `self-test: segmentation backend loaded on cuda ...`,
`self-test PASSED`, `Successful compile`, `BUILD COMPLETE`. Anything less is a
failure.

## 4. When it fails

See the table in PACKAGING.md section 6. The three that actually recur:

| Symptom | Fix |
|---|---|
| `No module named 'sam2'` from the exe | `SAM2_REPO` missing from `PYTHONPATH` for the Nuitka call, or `--include-package=sam2` dropped |
| Hydra/config error at first inference | `--include-package-data=sam2` dropped; the 13 `.yaml` configs were not bundled |
| `No SAM 2 checkpoint found` on the target | `weights\` not copied into `app.dist`, or code used `__file__` where it needs `sys.argv[0]` |

Anything that resolves a path inside the **install folder** must use
`sys.argv[0]`, not `__file__`: in a onefile build `__file__` is a temp
extraction directory. Use `refiner.paths.app_dir()`.

Diagnose root cause before editing `build.ps1`; don't paper over a failing
smoke test.

## 5. After a successful build

Report the installer path and size, and remind the user to run the acceptance
checklist in PACKAGING.md section 7 on a clean machine before handing it to the
vendor — this procedure cannot verify GPU throughput or real vendor hardware.
Do not commit `build\`, `dist\`, `.venv-build\` or `weights\` (all gitignored).
