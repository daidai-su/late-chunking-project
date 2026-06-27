"""Local validation that does not require pytest to be installed."""

from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def validate_notebook() -> None:
    import nbformat
    from nbformat.validator import validate

    for notebook_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        with notebook_path.open(encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
        validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{notebook_path}:cell-{index}", "exec")
        print(f"validated notebook: {notebook_path} ({len(notebook.cells)} cells)")


def run_test_functions() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    count = 0
    for test_path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(test_path.stem, test_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {test_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            value = getattr(module, name)
            if name.startswith("test_") and callable(value):
                value()
                count += 1
    print(f"direct test function checks passed: {count}")
    return count


def main() -> None:
    validate_notebook()
    count = run_test_functions()
    if count == 0:
        raise RuntimeError("No test functions were executed")


if __name__ == "__main__":
    main()
