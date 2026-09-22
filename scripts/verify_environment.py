from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

EXPECTED = {
    "numpy": "2.4.4",
    "scipy": "1.17.1",
    "sklearn": "1.8.0",
    "matplotlib": "3.10.9",
    "gymnasium": "1.3.0",
    "joblib": "1.5.3",
    "numba": "0.67.0",
    "tqdm": "4.70.1",
}

PROJECT_MODULES = [
    "dim_reduce",
    "distance_matrix",
    "kxx_matrix",
    "kxa_matrix",
    "kernel_regularized_least_squares",
    "autoencoder",
    "autoencoder_filtering_extended",
    "combine_multiple_autoencoders_extended",
    "parallel_autoencoders",
    "kernel_affine_hull_koopman_machines",
]

failed = False

print("=== PLATFORM ===")
print("Python executable:", sys.executable)
print("Python version:", platform.python_version())
print("Architecture:", platform.machine())
print()

if platform.machine() != "arm64":
    print("FAIL: expected arm64 architecture")
    failed = True
else:
    print("PASS: native arm64 architecture")

if sys.version_info[:2] != (3, 14):
    print(f"FAIL: expected Python 3.14, found {platform.python_version()}")
    failed = True
else:
    print("PASS: Python 3.14")

print()
print("=== THIRD-PARTY PACKAGES ===")

for module_name, expected_version in EXPECTED.items():
    try:
        module = importlib.import_module(module_name)
        actual = getattr(module, "__version__", None)
        if actual == expected_version:
            print(f"PASS: {module_name} {actual}")
        else:
            print(f"FAIL: {module_name}: expected {expected_version}, found {actual}")
            failed = True
    except Exception as exc:
        print(f"FAIL: {module_name}: {type(exc).__name__}: {exc}")
        failed = True

print()
print("=== PROJECT MODULES ===")

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

for module_name in PROJECT_MODULES:
    try:
        importlib.import_module(module_name)
        print(f"PASS: {module_name}")
    except Exception as exc:
        print(f"FAIL: {module_name}: {type(exc).__name__}: {exc}")
        failed = True

print()
print("=== RESULT ===")

if failed:
    print("ENVIRONMENT VERIFICATION FAILED")
    raise SystemExit(1)

print("ENVIRONMENT VERIFICATION PASSED")
