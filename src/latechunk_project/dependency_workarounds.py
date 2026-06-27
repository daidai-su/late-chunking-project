"""Small Colab dependency workarounds for text-only retrieval runs."""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Any


def remove_broken_torchcodec(python_executable: str = sys.executable) -> dict[str, Any]:
    """Remove a broken optional torchcodec install when it breaks text imports.

    Some Colab images can contain a torchcodec build that is incompatible with
    the available PyTorch or FFmpeg libraries. This project does not use audio
    or video decoding, so removing a broken torchcodec package is safer than
    blocking text-only retrieval baselines.
    """
    result: dict[str, Any] = {
        "package": "torchcodec",
        "needed_for_project": False,
        "present": None,
        "import_ok": None,
        "removed": False,
        "uninstall_returncode": None,
        "error": None,
    }

    try:
        importlib.import_module("torchcodec")
    except ModuleNotFoundError:
        result.update({"present": False, "import_ok": False})
        return result
    except Exception as exc:
        result.update({"present": True, "import_ok": False, "error": repr(exc)[:2000]})
        completed = subprocess.run(
            [python_executable, "-m", "pip", "uninstall", "-y", "torchcodec"],
            capture_output=True,
            text=True,
            check=False,
        )
        for module_name in list(sys.modules):
            if module_name == "torchcodec" or module_name.startswith("torchcodec."):
                sys.modules.pop(module_name, None)
        result.update(
            {
                "removed": completed.returncode == 0,
                "uninstall_returncode": completed.returncode,
            }
        )
        return result

    result.update({"present": True, "import_ok": True})
    return result

