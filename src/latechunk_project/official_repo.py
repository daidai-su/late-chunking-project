"""Thin wrappers around the official jina-ai/late-chunking repository."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    """Metadata for a completed subprocess with file-backed logs."""

    command: list[str]
    cwd: str
    returncode: int
    stdout_path: str
    stderr_path: str

    @property
    def command_text(self) -> str:
        return " ".join(self.command)


def clone_official_repo(
    repo_url: str,
    repo_dir: str | Path,
    force_rerun: bool = False,
) -> Path:
    """Clone the official repository if needed.

    The clone is intentionally shallow for Colab startup speed. Set force_rerun
    only when the caller explicitly wants to delete and recreate the clone.
    """
    target = Path(repo_dir)
    if force_rerun and target.exists():
        shutil.rmtree(target)

    if (target / ".git").exists():
        print(f"Official repo already present: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)
    return target


def install_official_repo(
    official_repo_dir: str | Path,
    python_executable: str = sys.executable,
) -> CommandResult:
    """Install the official repository in editable mode and log pip output."""
    repo_dir = Path(official_repo_dir)
    log_dir = repo_dir / ".install_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "pip_install_stdout.txt"
    stderr_path = log_dir / "pip_install_stderr.txt"
    command = [python_executable, "-m", "pip", "install", "-e", str(repo_dir)]
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        completed = subprocess.run(
            command,
            cwd=str(repo_dir),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        print(f"Editable install failed; see {stderr_path}. Continuing so raw logs are preserved.")
    return CommandResult(
        command=command,
        cwd=str(repo_dir),
        returncode=completed.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def official_baseline_command(task_name: str) -> list[str]:
    """Return the official baseline command path requested for Phase A."""
    return ["python", "run_chunked_eval.py", "--task-name", task_name]


def run_official_baseline(
    task_name: str,
    official_repo_dir: str | Path,
    output_dir: str | Path,
    python_executable: str = sys.executable,
    timeout_seconds: int | None = None,
) -> CommandResult:
    """Run the official chunked eval command and capture stdout/stderr logs."""
    repo_dir = Path(official_repo_dir)
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"official_{task_name}_stdout.txt"
    stderr_path = log_dir / f"official_{task_name}_stderr.txt"
    command = [python_executable, "run_chunked_eval.py", "--task-name", task_name]
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        completed = subprocess.run(
            command,
            cwd=str(repo_dir),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    return CommandResult(
        command=command,
        cwd=str(repo_dir),
        returncode=completed.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )

