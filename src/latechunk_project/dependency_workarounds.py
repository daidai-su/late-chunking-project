"""Small Colab dependency workarounds for text-only retrieval runs."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import subprocess
import sys
from typing import Any


def remove_broken_torchcodec(python_executable: str = sys.executable) -> dict[str, Any]:
    """Remove optional torchcodec for text-only retrieval runs.

    Some Colab images can contain a torchcodec build that is incompatible with
    the available PyTorch or FFmpeg libraries. A top-level ``import torchcodec``
    can still succeed while decoder shared libraries fail later during
    transformer imports. This project does not use audio or video decoding, so
    removing torchcodec when it is present is safer than blocking text-only
    retrieval baselines.
    """
    result: dict[str, Any] = {
        "package": "torchcodec",
        "needed_for_project": False,
        "present": None,
        "import_ok": None,
        "version": None,
        "removed": False,
        "uninstall_returncode": None,
        "error": None,
        "reason": "text_retrieval_does_not_need_audio_video_decoding",
    }

    try:
        result["version"] = metadata.version("torchcodec")
    except metadata.PackageNotFoundError:
        result.update({"present": False, "import_ok": False})
        return result

    result["present"] = True

    try:
        importlib.import_module("torchcodec")
        result["import_ok"] = True
    except Exception as exc:
        result.update({"import_ok": False, "error": repr(exc)[:2000]})

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
