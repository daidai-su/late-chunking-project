"""Generate the Phase B aggregation experiment Colab notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "02_late_chunking_aggregation_experiment_colab.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "colab": {"name": "02_late_chunking_aggregation_experiment_colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb.cells = [
        markdown(
            """
            # Late Chunking Aggregation Experiment

            Phase B evaluates training-free chunk-to-document aggregation over
            chunk-level Late Chunking rankings. The model/evaluation work runs
            in a fresh Python subprocess so Colab notebook-kernel NumPy state
            cannot leak into MTEB imports.
            """
        ),
        code(
            '''
            PROJECT_NAME = "late_chunking_aggregation"
            PUBLIC_REPO_URL = "https://github.com/daidai-su/late-chunking-project.git"
            OFFICIAL_REPO_URL = "https://github.com/jina-ai/late-chunking.git"

            MODE = "small"  # smoke / small / full
            TASKS = ["SciFactChunked"]
            OPTIONAL_TASKS = ["NFCorpusChunked"]

            SEED = 42
            USE_GPU_IF_AVAILABLE = True
            FORCE_RERUN = False
            CONFIRM_FULL_RUN = False

            OUTPUT_DIR = "/content/late_chunking_outputs"
            AGG_OUTPUT_SUBDIR = "aggregation_experiment"
            OFFICIAL_REPO_DIR = "/content/official_late_chunking"

            MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"
            CHUNKING_STRATEGY = "fixed"
            CHUNK_SIZE = 256
            N_SENTENCES = 5
            BATCH_SIZE = 1

            CHUNK_TOP_K_FOR_AGGREGATION = 1000
            DOC_TOP_K_EVAL = [10, 100]

            TOPK_MEAN_K = 3
            SOFTMAX_TOPK_K = 3
            SOFTMAX_TAU = 0.05
            RRF_K = 60

            RUN_BM25 = True
            RUN_RRF_FUSION = True
            UPDATE_PROJECT_ON_START = True
            '''
        ),
        markdown("## Install Project Dependencies"),
        code(
            r'''
            import importlib
            import os
            import subprocess
            import sys
            from pathlib import Path

            def run_command(command, cwd=None, check=True):
                print("+", " ".join(str(part) for part in command))
                return subprocess.run(command, cwd=cwd, check=check)

            def maybe_update_project(candidate):
                if UPDATE_PROJECT_ON_START and (candidate / ".git").exists():
                    run_command(["git", "-C", str(candidate), "pull", "--ff-only"], check=False)
                return candidate

            def find_or_clone_project():
                current = Path.cwd()
                candidates = [
                    current,
                    current.parent if current.name == "notebooks" else current,
                    Path("/content") / "late_chunking_project",
                    Path("/content") / PROJECT_NAME,
                ]
                for candidate in candidates:
                    if (candidate / "src" / "latechunk_project").exists():
                        return maybe_update_project(candidate)

                target = Path("/content") / "late_chunking_project"
                if not target.exists():
                    run_command(["git", "clone", PUBLIC_REPO_URL, str(target)])
                return maybe_update_project(target)

            PROJECT_ROOT = find_or_clone_project()
            os.chdir(PROJECT_ROOT)
            print(f"Project root: {PROJECT_ROOT}")

            src_path = str(PROJECT_ROOT / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

            requirements_path = PROJECT_ROOT / "requirements-colab.txt"
            if requirements_path.exists():
                run_command([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)])

            if (PROJECT_ROOT / "pyproject.toml").exists() or (PROJECT_ROOT / "setup.py").exists():
                editable_install = run_command(
                    [sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT)],
                    check=False,
                )
                if editable_install.returncode != 0:
                    print("Editable install failed; continuing because src/ is on sys.path.")

            import latechunk_project
            print(f"latechunk_project import OK: {latechunk_project.__version__}")

            import latechunk_project.dependency_workarounds as dependency_workarounds

            dependency_workarounds = importlib.reload(dependency_workarounds)
            ensure_numpy_stack_healthy = dependency_workarounds.ensure_numpy_stack_healthy
            remove_optional_media_dependencies = dependency_workarounds.remove_optional_media_dependencies

            numpy_stack_status = ensure_numpy_stack_healthy(check_current_process=False)
            print("On-disk NumPy stack check:", numpy_stack_status)
            dependency_workaround_events = remove_optional_media_dependencies()
            print("Optional media dependency checks:", dependency_workaround_events)
            '''
        ),
        markdown("## Clone Official Repository"),
        code(
            '''
            import importlib
            import sys
            from pathlib import Path

            import latechunk_project.dependency_workarounds as dependency_workarounds

            dependency_workarounds = importlib.reload(dependency_workarounds)
            ensure_numpy_stack_healthy = dependency_workarounds.ensure_numpy_stack_healthy
            remove_optional_media_dependencies = dependency_workarounds.remove_optional_media_dependencies
            from latechunk_project.official_repo import clone_official_repo, install_official_repo
            from latechunk_project.run_utils import ensure_output_tree

            output_dirs = ensure_output_tree(OUTPUT_DIR)
            AGG_OUTPUT_DIR = Path(OUTPUT_DIR) / AGG_OUTPUT_SUBDIR
            for name in ["rankings", "tables", "figures", "logs"]:
                (AGG_OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

            official_repo_path = clone_official_repo(
                OFFICIAL_REPO_URL,
                OFFICIAL_REPO_DIR,
                force_rerun=FORCE_RERUN,
            )
            official_install = install_official_repo(official_repo_path)
            numpy_stack_status_after_official_install = ensure_numpy_stack_healthy(check_current_process=False)
            dependency_workaround_events.extend(remove_optional_media_dependencies())

            print(f"Official repo: {official_repo_path}")
            print(f"Official editable install return code: {official_install.returncode}")
            print(f"Official install stdout: {official_install.stdout_path}")
            print(f"Official install stderr: {official_install.stderr_path}")
            print("On-disk NumPy stack check after official install:", numpy_stack_status_after_official_install)
            print(f"Aggregation outputs: {AGG_OUTPUT_DIR}")
            '''
        ),
        markdown("## Run Aggregation Experiment"),
        code(
            '''
            import json
            import subprocess
            import sys
            from pathlib import Path

            if MODE == "full" and not CONFIRM_FULL_RUN:
                raise SystemExit("Set CONFIRM_FULL_RUN = True before running full mode.")

            AGG_OUTPUT_DIR = Path(OUTPUT_DIR) / AGG_OUTPUT_SUBDIR
            logs_dir = AGG_OUTPUT_DIR / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            runner_config = {
                "project_root": str(PROJECT_ROOT),
                "official_repo_path": str(official_repo_path),
                "output_dir": OUTPUT_DIR,
                "agg_output_subdir": AGG_OUTPUT_SUBDIR,
                "mode": MODE,
                "tasks": TASKS,
                "optional_tasks": OPTIONAL_TASKS,
                "seed": SEED,
                "use_gpu_if_available": USE_GPU_IF_AVAILABLE,
                "confirm_full_run": CONFIRM_FULL_RUN,
                "model_name": MODEL_NAME,
                "chunking_strategy": CHUNKING_STRATEGY,
                "chunk_size": CHUNK_SIZE,
                "n_sentences": N_SENTENCES,
                "batch_size": BATCH_SIZE,
                "chunk_top_k_for_aggregation": CHUNK_TOP_K_FOR_AGGREGATION,
                "doc_top_k_eval": DOC_TOP_K_EVAL,
                "topk_mean_k": TOPK_MEAN_K,
                "softmax_topk_k": SOFTMAX_TOPK_K,
                "softmax_tau": SOFTMAX_TAU,
                "rrf_k": RRF_K,
                "run_bm25": RUN_BM25,
                "run_rrf_fusion": RUN_RRF_FUSION,
                "dependency_workaround_events": dependency_workaround_events,
            }
            RUNNER_CONFIG_PATH = AGG_OUTPUT_DIR / "aggregation_runner_config.json"
            RUNNER_CONFIG_PATH.write_text(json.dumps(runner_config, indent=2, sort_keys=True), encoding="utf-8")

            stdout_path = logs_dir / "aggregation_runner_stdout.txt"
            stderr_path = logs_dir / "aggregation_runner_stderr.txt"
            command = [
                sys.executable,
                "-m",
                "latechunk_project.aggregation_experiment",
                "--config-json",
                str(RUNNER_CONFIG_PATH),
            ]
            print("+", " ".join(str(part) for part in command))
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_file:
                completed = subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    check=False,
                )

            def print_tail(path, max_chars=4000):
                path = Path(path)
                print(f"--- tail: {path} ---")
                if path.exists():
                    print(path.read_text(encoding="utf-8")[-max_chars:])
                else:
                    print("missing")

            print(f"Aggregation runner return code: {completed.returncode}")
            print_tail(stdout_path)
            if completed.returncode != 0:
                print_tail(stderr_path)
                raise RuntimeError(f"Aggregation runner failed; see {stderr_path}")

            AGG_MANIFEST_PATH = AGG_OUTPUT_DIR / "aggregation_run_manifest.json"
            aggregation_manifest = json.loads(AGG_MANIFEST_PATH.read_text(encoding="utf-8"))
            summary_path = Path(aggregation_manifest["output_file_paths"]["aggregation_results_summary_csv"])
            print(f"Aggregation manifest: {AGG_MANIFEST_PATH}")
            print(f"Aggregation summary: {summary_path}")
            if summary_path.exists():
                print(summary_path.read_text(encoding="utf-8")[:5000])
            aggregation_manifest
            '''
        ),
        markdown("## Results Reminder"),
        markdown(
            """
            Do not report improvements unless this notebook actually executed and
            saved the result artifacts. BM25/RRF rows are hybrid dense + lexical
            baselines, not pure Late Chunking aggregation improvements.
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
