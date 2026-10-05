"""The CLI surface must stay in step with the available backends.

A backend added to the factory but not to argparse is invisible to users, which is
exactly how --backend sam2 shipped broken.
"""

import sys

import pytest

import app as app_module
from refiner.backend.fallback import FallbackBackend
from refiner.backend.sam2_backend import Sam2Backend
from refiner.backend.sam3_backend import Sam3Backend
from refiner.ui.workers import build_backend

BACKENDS = ["sam2", "sam3", "fallback"]


@pytest.fixture
def argv(monkeypatch):
    def _set(*args):
        monkeypatch.setattr(sys, "argv", ["app.py", *args])
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
