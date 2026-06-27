"""Environment inspection helpers for Colab and local smoke checks."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _package_version(distribution_name: str, module_name: str | None = None) -> str | None:
    """Return a package version without importing heavyweight packages unless needed."""
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        if module_name is None:
            return None

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "installed_version_unknown"))


def get_git_commit(path: str | os.PathLike[str] | None = None) -> str | None:
    """Return the current git commit for a path, or None when unavailable."""
    repo_path = Path(path or ".")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _torch_info(use_gpu_if_available: bool = True) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "device": "cpu",
    }
    try:
        import torch
    except Exception:
        return info

    info["torch_version"] = getattr(torch, "__version__", "installed_version_unknown")
    cuda_available = bool(torch.cuda.is_available())
    info["cuda_available"] = cuda_available
    if cuda_available:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["gpu_name"] = "cuda_device_name_unavailable"
    if use_gpu_if_available and cuda_available:
        info["device"] = "cuda"
    return info


def collect_environment_info(
    project_path: str | os.PathLike[str] | None = None,
    official_repo_path: str | os.PathLike[str] | None = None,
    use_gpu_if_available: bool = True,
) -> dict[str, Any]:
    """Collect the environment fields requested by the Phase A notebook."""
    info: dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "current_working_directory": os.getcwd(),
        "project_git_commit": get_git_commit(project_path or "."),
        "official_repo_git_commit": get_git_commit(official_repo_path)
        if official_repo_path is not None and Path(official_repo_path).exists()
        else None,
        "package_versions": {
            "transformers": _package_version("transformers"),
            "sentence-transformers": _package_version("sentence-transformers"),
            "faiss": _package_version("faiss-cpu", "faiss"),
            "beir": _package_version("beir"),
            "mteb": _package_version("mteb"),
        },
    }
    info.update(_torch_info(use_gpu_if_available=use_gpu_if_available))
    return info


def print_environment_info(info: dict[str, Any]) -> None:
    """Print environment details in a stable, notebook-friendly order."""
    print(f"Python version: {info.get('python_version')}")
    print(f"Platform: {info.get('platform')}")
    print(f"Torch version: {info.get('torch_version')}")
    print(f"CUDA available: {info.get('cuda_available')}")
    print(f"GPU name: {info.get('gpu_name')}")
    versions = info.get("package_versions", {})
    print(f"Transformers version: {versions.get('transformers')}")
    print(f"Sentence-transformers version: {versions.get('sentence-transformers')}")
    print(f"FAISS version: {versions.get('faiss')}")
    print(f"BEIR version: {versions.get('beir')}")
    print(f"Current working directory: {info.get('current_working_directory')}")
    print(f"Project git commit: {info.get('project_git_commit')}")
    print(f"Official late-chunking git commit: {info.get('official_repo_git_commit')}")

