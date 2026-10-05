from __future__ import annotations

import importlib


def check(name: str):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "installed")
        print(f"[OK] {name}: {version}")
        return module
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return None


print("Filter Annotation Refiner environment check\n")
check("PySide6")
check("numpy")
check("cv2")
check("einops")
torch = check("torch")
if torch is not None:
    print(f"     CUDA available: {torch.cuda.is_available()}")
    print(f"     Torch CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
check("sam3")
