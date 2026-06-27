"""Small Colab dependency workarounds for text-only retrieval runs."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import subprocess
import sys
from typing import Any


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
