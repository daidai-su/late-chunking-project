"""Generate the Colab notebook with nbformat."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_late_chunking_baseline_colab.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "colab": {"name": "01_late_chunking_baseline_colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb.cells = [
        markdown(
            """
            # Late Chunking Official Baseline

            This notebook prepares a reproducible Phase A run for the official
            `jina-ai/late-chunking` baseline. It does not report scores unless
            the commands below actually execute and save logs.
            """
        ),
        code(
            '''
            PROJECT_NAME = "late_chunking_project"
            PUBLIC_REPO_URL = "https://github.com/<YOUR_GITHUB_USERNAME>/late-chunking-project.git"
            OFFICIAL_REPO_URL = "https://github.com/jina-ai/late-chunking.git"

            MODE = "smoke"  # smoke / small / full
            TASKS = ["SciFactChunked"]  # optional later: ["SciFactChunked", "NFCorpusChunked"]
            SEED = 42

            USE_GPU_IF_AVAILABLE = True
            FORCE_RERUN = False

            OUTPUT_DIR = "/content/late_chunking_outputs"
            OFFICIAL_REPO_DIR = "/content/official_late_chunking"

            # For smoke mode only
            SMOKE_MAX_QUERIES = 20
            SMOKE_MAX_CORPUS_DOCS = 500

            # For official reproduction
            RUN_OFFICIAL_BASELINE = True
            RUN_INTERNAL_BASELINE_IF_AVAILABLE = True

            CONFIRM_FULL_RUN = False
            '''
        ),
        markdown("## Install Project Dependencies"),
        code(
            r'''
            import os
            import subprocess
            import sys
            from pathlib import Path

            def run_command(command, cwd=None, check=True):
                print("+", " ".join(str(part) for part in command))
                return subprocess.run(command, cwd=cwd, check=check)

            def find_or_clone_project():
                current = Path.cwd()
                candidates = [
                    current,
                    current.parent if current.name == "notebooks" else current,
                    Path("/content") / PROJECT_NAME,
                ]
                for candidate in candidates:
                    if (candidate / "src" / "latechunk_project").exists():
                        return candidate

                target = Path("/content") / PROJECT_NAME
                if "<YOUR_GITHUB_USERNAME>" in PUBLIC_REPO_URL:
                    raise RuntimeError(
                        "Edit PUBLIC_REPO_URL before running this notebook standalone in Colab, "
                        "or clone the project repository before running."
                    )
                if not target.exists():
                    run_command(["git", "clone", PUBLIC_REPO_URL, str(target)])
                return target

            PROJECT_ROOT = find_or_clone_project()
            os.chdir(PROJECT_ROOT)
            print(f"Project root: {PROJECT_ROOT}")

            src_path = str(PROJECT_ROOT / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            print(f"Added to Python path: {src_path}")

            requirements_path = PROJECT_ROOT / "requirements-colab.txt"
            if requirements_path.exists():
                run_command([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)])

            if (PROJECT_ROOT / "pyproject.toml").exists() or (PROJECT_ROOT / "setup.py").exists():
                editable_install = run_command(
                    [sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT)],
                    check=False,
                )
                if editable_install.returncode != 0:
                    print(
                        "Editable install failed, but src/ is already on sys.path. "
                        "Continuing with direct source imports."
                    )

            import latechunk_project
            print(f"latechunk_project import OK: {latechunk_project.__version__}")

            from latechunk_project.dependency_workarounds import remove_optional_media_dependencies

            dependency_workaround_events = []
            optional_media_status = remove_optional_media_dependencies()
            dependency_workaround_events.extend(optional_media_status)
            print("Optional media dependency checks:", optional_media_status)
            '''
        ),
        markdown("## Clone And Install The Official Repository"),
        code(
            '''
            import sys
            from pathlib import Path

            if "PROJECT_ROOT" not in globals():
                raise RuntimeError("Run the 'Install Project Dependencies' cell before this cell.")

            src_path = str(Path(PROJECT_ROOT) / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

            from latechunk_project.official_repo import clone_official_repo, install_official_repo
            from latechunk_project.run_utils import ensure_output_tree
            from latechunk_project.dependency_workarounds import remove_optional_media_dependencies

            output_dirs = ensure_output_tree(OUTPUT_DIR)
            official_repo_path = clone_official_repo(
                OFFICIAL_REPO_URL,
                OFFICIAL_REPO_DIR,
                force_rerun=FORCE_RERUN,
            )
            official_install = install_official_repo(official_repo_path)
            optional_media_status_after_official_install = remove_optional_media_dependencies()
            if "dependency_workaround_events" not in globals():
                dependency_workaround_events = []
            dependency_workaround_events.extend(optional_media_status_after_official_install)
            print(f"Official repo: {official_repo_path}")
            print(f"Official editable install return code: {official_install.returncode}")
            print(f"Official install stdout: {official_install.stdout_path}")
            print(f"Official install stderr: {official_install.stderr_path}")
            print("Optional media checks after official install:", optional_media_status_after_official_install)
            '''
        ),
        markdown("## Environment Check"),
        code(
            '''
            from latechunk_project.env_utils import collect_environment_info, print_environment_info

            env_info = collect_environment_info(
                project_path=PROJECT_ROOT,
                official_repo_path=official_repo_path,
                use_gpu_if_available=USE_GPU_IF_AVAILABLE,
            )
            print_environment_info(env_info)
            '''
        ),
        markdown("## Manifest Setup"),
        code(
            '''
            from latechunk_project.official_repo import official_baseline_command
            from latechunk_project.run_utils import save_run_manifest, utc_now_iso

            manifest = {
                "start_timestamp": utc_now_iso(),
                "end_timestamp": None,
                "project_git_commit": env_info.get("project_git_commit"),
                "official_repo_git_commit": env_info.get("official_repo_git_commit"),
                "task_names": TASKS,
                "mode": MODE,
                "seed": SEED,
                "device": env_info.get("device"),
                "gpu_name": env_info.get("gpu_name"),
                "package_versions": env_info.get("package_versions"),
                "official_commands": {
                    task: " ".join(official_baseline_command(task)) for task in TASKS
                },
                "output_file_paths": {},
                "metrics_parsed_successfully": False,
                "metrics": {},
                "notes": [],
                "dependency_workarounds": dependency_workaround_events
                if "dependency_workaround_events" in globals()
                else [],
            }
            save_run_manifest(OUTPUT_DIR, manifest)
            print(f"Manifest initialized at {OUTPUT_DIR}/run_manifest.json")
            '''
        ),
        markdown("## Smoke Checks"),
        code(
            '''
            from latechunk_project.metrics_utils import compute_ndcg_at_k, compute_recall_at_k
            from latechunk_project.result_parsing import parse_official_metrics

            toy_qrels = {"q1": {"d1": 1, "d2": 1}}
            toy_rankings = {"q1": ["d1", "d3", "d2"]}
            print("Toy nDCG@10:", compute_ndcg_at_k(toy_rankings, toy_qrels, k=10))
            print("Toy Recall@2:", compute_recall_at_k(toy_rankings, toy_qrels, k=2))
            print("Parser smoke:", parse_official_metrics("nDCG@10: 0.1234"))

            if RUN_INTERNAL_BASELINE_IF_AVAILABLE:
                print("Internal baseline hook is present, but no internal baseline is implemented in Phase A.")
            '''
        ),
        markdown("## Official Baseline"),
        code(
            '''
            import csv
            from pathlib import Path

            from latechunk_project.official_repo import run_official_baseline
            from latechunk_project.result_parsing import parse_metrics_from_file
            from latechunk_project.run_utils import save_run_manifest, utc_now_iso

            if MODE == "full" and not CONFIRM_FULL_RUN:
                manifest["end_timestamp"] = utc_now_iso()
                manifest["notes"].append("Full mode requested without CONFIRM_FULL_RUN=True; stopped before heavy work.")
                save_run_manifest(OUTPUT_DIR, manifest)
                raise SystemExit("Set CONFIRM_FULL_RUN = True before running full mode.")

            should_run_official = RUN_OFFICIAL_BASELINE and MODE in {"small", "full"}
            task_results = []

            for task in TASKS:
                stdout_path = Path(OUTPUT_DIR) / "logs" / f"official_{task}_stdout.txt"
                stderr_path = Path(OUTPUT_DIR) / "logs" / f"official_{task}_stderr.txt"

                if should_run_official:
                    print(f"Running official baseline for {task}")
                    result = run_official_baseline(
                        task_name=task,
                        official_repo_dir=official_repo_path,
                        output_dir=OUTPUT_DIR,
                    )
                    returncode = result.returncode
                    stdout_path = Path(result.stdout_path)
                    stderr_path = Path(result.stderr_path)
                else:
                    command_text = f"python run_chunked_eval.py --task-name {task}"
                    message = (
                        f"MODE={MODE}: full official command was not launched. "
                        f"Switch MODE='small' to run: {command_text}\\n"
                    )
                    stdout_path.write_text(message, encoding="utf-8")
                    stderr_path.write_text("", encoding="utf-8")
                    print(message)
                    manifest["notes"].append(message.strip())
                    returncode = None

                metrics = {}
                metrics.update(parse_metrics_from_file(stdout_path))
                metrics.update(parse_metrics_from_file(stderr_path))
                parsed = bool(metrics)

                manifest["output_file_paths"][task] = {
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                }
                manifest["metrics"][task] = metrics
                task_results.append(
                    {
                        "task": task,
                        "returncode": returncode,
                        "metrics_parsed": parsed,
                        **metrics,
                    }
                )

            manifest["metrics_parsed_successfully"] = any(bool(item.get("metrics_parsed")) for item in task_results)
            manifest["end_timestamp"] = utc_now_iso()

            metrics_table = Path(OUTPUT_DIR) / "tables" / "metrics_summary.csv"
            if task_results:
                fieldnames = sorted({key for row in task_results for key in row})
                with metrics_table.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(task_results)
                manifest["output_file_paths"]["metrics_summary_csv"] = str(metrics_table)

            save_run_manifest(OUTPUT_DIR, manifest)
            print(f"Saved manifest: {OUTPUT_DIR}/run_manifest.json")
            print(f"Saved metrics table: {metrics_table}")
            task_results
            '''
        ),
        markdown("## Parse Official Result JSON Artifacts"),
        code(
            '''
            import csv
            from pathlib import Path

            from latechunk_project.result_parsing import summarize_mteb_result_jsons
            from latechunk_project.run_utils import save_run_manifest, utc_now_iso

            result_rows, result_metrics = summarize_mteb_result_jsons(official_repo_path, TASKS)

            if result_rows:
                summary_path = Path(OUTPUT_DIR) / "tables" / "official_results_summary.csv"
                fieldnames = [
                    "task",
                    "method",
                    "main_score",
                    "ndcg_at_10",
                    "recall_at_10",
                    "map_at_10",
                    "mrr_at_10",
                    "evaluation_time",
                    "result_json",
                ]
                with summary_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(result_rows)

                manifest["metrics"].update(result_metrics)
                manifest["metrics_parsed_successfully"] = True
                manifest["output_file_paths"]["official_results_summary_csv"] = str(summary_path)
                official_result_jsons = {task: {} for task in TASKS}
                for row in result_rows:
                    if row["task"] in official_result_jsons:
                        official_result_jsons[row["task"]][row["method"]] = row["result_json"]
                manifest["output_file_paths"]["official_result_jsons"] = official_result_jsons
                manifest["end_timestamp"] = utc_now_iso()
                save_run_manifest(OUTPUT_DIR, manifest)

                print(f"Saved official results summary: {summary_path}")
                for row in result_rows:
                    print(row)
            else:
                print("No official MTEB result JSON files found yet.")
                print("If the official command returned 0, inspect the official repo result directories.")
            '''
        ),
        markdown(
            """
            ## After The Run

            Check `run_manifest.json`, `official_results_summary.csv`, and the
            raw stdout/stderr logs before reporting any result. If parsing fails,
            use the raw official result JSON files as the source of truth.
            """
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
