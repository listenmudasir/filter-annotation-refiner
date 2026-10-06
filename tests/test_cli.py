"""The CLI surface must stay in step with the available backends.

A backend added to the factory but not to argparse is invisible to users, which is
exactly how --backend sam2 shipped broken.
"""

import sys

import pytest

from refiner import cli as app_module
from refiner.backend.fallback import FallbackBackend
from refiner.backend.sam2_backend import Sam2Backend
from refiner.backend.sam3_backend import Sam3Backend
from refiner.ui.workers import build_backend

BACKENDS = ["sam2", "sam3", "fallback"]


@pytest.fixture
def argv(monkeypatch):
    def _set(*args):
        monkeypatch.setattr(sys, "argv", ["filter-annotation-refiner", *args])
    return _set


@pytest.mark.parametrize("name", BACKENDS)
def test_cli_accepts_every_backend(argv, name):
    argv("--backend", name)
    assert app_module.parse_args().backend == name


def test_cli_rejects_unknown_backend(argv):
    argv("--backend", "nope")
    with pytest.raises(SystemExit):
        app_module.parse_args()


def test_cli_default_is_a_real_model_not_the_placeholder(argv):
    argv()
    default = app_module.parse_args().backend
    assert default != "fallback", "the default backend must produce real segmentation"
    assert default in BACKENDS


def test_cli_choices_match_the_factory(argv):
    """Every advertised choice must actually be constructible."""
    for name in BACKENDS:
        argv("--backend", name)
        assert app_module.parse_args().backend == name
        assert build_backend(name) is not None


@pytest.mark.parametrize(
    "name,expected",
    [("fallback", FallbackBackend), ("sam2", Sam2Backend), ("sam3", Sam3Backend)],
)
def test_factory_builds_each_backend(name, expected):
    """Construction must not require a GPU or a checkpoint; load() does that."""
    assert isinstance(build_backend(name, device="cpu"), expected)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        build_backend("not-a-backend")


def test_cli_passes_device_and_checkpoint_through(argv):
    argv("--backend", "sam2", "--device", "cuda", "--checkpoint", "/tmp/x.pt")
    args = app_module.parse_args()
    assert (args.backend, args.device, args.checkpoint) == ("sam2", "cuda", "/tmp/x.pt")


def test_main_window_default_backend_matches_cli(argv):
    """The GUI default and the CLI default must not drift apart."""
    import inspect

    from refiner.ui.main_window import MainWindow

    argv()
    cli_default = app_module.parse_args().backend
    gui_default = inspect.signature(MainWindow.__init__).parameters["backend_name"].default
    assert gui_default == cli_default


# --------------------------------------------------------------------------- #
# SAM 2 discovery
# --------------------------------------------------------------------------- #


def make_checkout(root, name: str):
    """A directory that looks like a SAM 2 clone."""
    repo = root / name
    (repo / "sam2").mkdir(parents=True)
    (repo / "sam2" / "__init__.py").write_text("")
    return repo


def test_discovery_matches_renamed_forks(tmp_path, monkeypatch):
    """A clone named sam2_cell-main or sam2-main must be found, not just 'sam2'."""
    from refiner.backend import sam2_backend as sb

    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    for name in ("sam2_cell-main", "sam2-main", "segment-anything-2-main"):
        make_checkout(desktop, name)

    monkeypatch.delenv("SAM2_REPO", raising=False)
    monkeypatch.setattr(sb, "_SEARCH_PARENTS", (str(desktop),))
    found = {p.name for p in sb._candidate_repos() if sb._looks_like_checkout(p)}
    assert found == {"sam2_cell-main", "sam2-main", "segment-anything-2-main"}


def test_env_var_takes_precedence(tmp_path, monkeypatch):
    from refiner.backend import sam2_backend as sb

    explicit = make_checkout(tmp_path, "mine")
    other = tmp_path / "other"
    other.mkdir()
    make_checkout(other, "sam2")

    monkeypatch.setenv("SAM2_REPO", str(explicit))
    monkeypatch.setattr(sb, "_SEARCH_PARENTS", (str(other),))
    assert sb._candidate_repos()[0] == explicit


def test_non_checkout_directories_are_ignored(tmp_path, monkeypatch):
    from refiner.backend import sam2_backend as sb

    base = tmp_path / "d"
    (base / "sam2_notes").mkdir(parents=True)  # name matches, but no package
    monkeypatch.delenv("SAM2_REPO", raising=False)
    monkeypatch.setattr(sb, "_SEARCH_PARENTS", (str(base),))
    assert [p for p in sb._candidate_repos() if sb._looks_like_checkout(p)] == []


def test_missing_sam2_names_what_was_searched(tmp_path, monkeypatch):
    from refiner.backend import sam2_backend as sb

    monkeypatch.delenv("SAM2_REPO", raising=False)
    monkeypatch.setattr(sb, "_SEARCH_PARENTS", (str(tmp_path),))
    monkeypatch.setattr(sb.importlib, "import_module", _raise_import_error)
    with pytest.raises(RuntimeError) as excinfo:
        sb._ensure_sam2_importable()
    message = str(excinfo.value)
    assert "pip install" in message
    assert "SAM2_REPO" in message
    assert str(tmp_path) in message, "the error must say where it looked"


def _raise_import_error(name, *a, **k):
    raise ImportError(name)


def test_version_declarations_agree():
    """VERSION and pyproject drifted (1.1.0 vs 1.0.0); a release needs one number."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    version_file = (root / "VERSION").read_text(encoding="utf-8").strip()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert version_file == pyproject["project"]["version"]


def test_console_script_is_declared():
    """pip install must expose a runnable command, not just the package."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["filter-annotation-refiner"] == "refiner.cli:main"
    # And every runtime dependency is declared, not only in requirements.txt.
    deps = " ".join(pyproject["project"]["dependencies"]).lower()
    for package in ("pyside6", "numpy", "pillow", "opencv", "pyyaml"):
        assert package in deps, package


def test_every_package_subdir_is_shipped():
    """A missing entry here silently omits a subpackage from the wheel."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    declared = set(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
                   ["tool"]["setuptools"]["packages"])
    found = {"refiner"} | {
        f"refiner.{p.name}" for p in (root / "refiner").iterdir()
        if p.is_dir() and (p / "__init__.py").exists()
    }
    assert found <= declared, f"not shipped: {found - declared}"
