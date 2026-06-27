"""Small Colab dependency workarounds for text-only retrieval runs."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import subprocess
import sys
from typing import Any


NUMPY_STACK_REQUIREMENTS = [
    "numpy==1.26.4",
    "scipy>=1.12,<1.15",
    "scikit-learn>=1.4,<1.6",
    "pandas>=2.2,<2.3",
]


NUMPY_STACK_CHECK_CODE = r"""
import numpy as np
print("numpy", np.__version__)
print("numpy_char", np.char.lower(["ABC"])[0])
from numpy.random import RandomState
print("numpy_random", RandomState(0).rand(1)[0])
import scipy
print("scipy", scipy.__version__)
import sklearn
print("sklearn", sklearn.__version__)
import pandas
print("pandas", pandas.__version__)
"""


def ensure_numpy_stack_healthy(
    python_executable: str = sys.executable,
    repair: bool = True,
) -> dict[str, Any]:
    """Check and repair the NumPy/SciPy/sklearn stack before importing MTEB.

    The check runs in a subprocess so a broken NumPy import does not poison the
    current notebook kernel. If NumPy is already imported in the current process
    and the subprocess check fails, the function raises because a restart is the
    only reliable recovery.
    """
    result: dict[str, Any] = {
        "check": "numpy_stack",
        "requirements": NUMPY_STACK_REQUIREMENTS,
        "numpy_already_imported": "numpy" in sys.modules,
        "initial_returncode": None,
        "initial_stdout": None,
        "initial_stderr": None,
        "repair_attempted": False,
        "uninstall_returncode": None,
        "repair_returncode": None,
        "final_returncode": None,
        "final_stdout": None,
        "final_stderr": None,
        "healthy": False,
    }

    initial = subprocess.run(
        [python_executable, "-c", NUMPY_STACK_CHECK_CODE],
        capture_output=True,
        text=True,
        check=False,
    )
    result.update(
        {
            "initial_returncode": initial.returncode,
            "initial_stdout": initial.stdout[-2000:],
            "initial_stderr": initial.stderr[-2000:],
        }
    )
    if initial.returncode == 0:
        result.update(
            {
                "final_returncode": initial.returncode,
                "final_stdout": initial.stdout[-2000:],
                "final_stderr": initial.stderr[-2000:],
                "healthy": True,
            }
        )
        return result

    if result["numpy_already_imported"]:
        raise RuntimeError(
            "NumPy stack is broken and numpy is already imported in this kernel. "
            "Restart the runtime after dependency installation."
        )

    if repair:
        result["repair_attempted"] = True
        uninstall_command = [
            python_executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "numpy",
            "scipy",
            "scikit-learn",
            "pandas",
        ]
        uninstalled = subprocess.run(
            uninstall_command,
            capture_output=True,
            text=True,
            check=False,
        )
        result["uninstall_returncode"] = uninstalled.returncode
        repair_command = [
            python_executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "--force-reinstall",
            *NUMPY_STACK_REQUIREMENTS,
        ]
        repaired = subprocess.run(
            repair_command,
            capture_output=True,
            text=True,
            check=False,
        )
        result["repair_returncode"] = repaired.returncode

    final = subprocess.run(
        [python_executable, "-c", NUMPY_STACK_CHECK_CODE],
        capture_output=True,
        text=True,
        check=False,
    )
    result.update(
        {
            "final_returncode": final.returncode,
            "final_stdout": final.stdout[-2000:],
            "final_stderr": final.stderr[-2000:],
            "healthy": final.returncode == 0,
        }
    )
    if final.returncode != 0:
        raise RuntimeError(
            "NumPy stack check failed after repair. "
            f"stderr tail: {final.stderr[-1000:]}"
        )
    return result


def _remove_optional_package(package_name: str, python_executable: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "package": package_name,
        "needed_for_project": False,
        "present": None,
        "import_ok": None,
        "version": None,
        "removed": False,
        "uninstall_returncode": None,
        "error": None,
        "reason": "text_retrieval_does_not_need_audio_video_image_decoding",
    }

    try:
        result["version"] = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        result.update({"present": False, "import_ok": False})
        return result

    result["present"] = True

    try:
        importlib.import_module(package_name)
        result["import_ok"] = True
    except Exception as exc:
        result.update({"import_ok": False, "error": repr(exc)[:2000]})

    completed = subprocess.run(
        [python_executable, "-m", "pip", "uninstall", "-y", package_name],
        capture_output=True,
        text=True,
        check=False,
    )
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)
    result.update(
        {
            "removed": completed.returncode == 0,
            "uninstall_returncode": completed.returncode,
        }
    )
    return result


def remove_optional_media_dependencies(python_executable: str = sys.executable) -> list[dict[str, Any]]:
    """Remove optional audio/video/image packages for text-only retrieval.

    Some Colab images can contain a torchcodec build that is incompatible with
    the available PyTorch or FFmpeg libraries. A top-level ``import torchcodec``
    can still succeed while decoder shared libraries fail later during
    transformer imports. This project does not use audio or video decoding, so
    removing torchcodec when it is present is safer than blocking text-only
    retrieval baselines.

    The same applies to torchvision in Colab runtimes where torchvision and
    torch are ABI-incompatible. Transformers can try to import torchvision for
    image helpers even though this project only runs text retrieval.
    """
    return [
        _remove_optional_package("torchcodec", python_executable),
        _remove_optional_package("torchvision", python_executable),
    ]


def remove_broken_torchcodec(python_executable: str = sys.executable) -> dict[str, Any]:
    """Backward-compatible wrapper for older notebook cells."""
    return _remove_optional_package("torchcodec", python_executable)
