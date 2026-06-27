from latechunk_project import dependency_workarounds as workarounds


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_numpy_stack_check_reports_current_process_success():
    original_run = workarounds.subprocess.run
    original_current_check = workarounds._check_numpy_stack_in_current_process
    original_loaded_modules = workarounds._loaded_numpy_stack_modules
    try:
        workarounds.subprocess.run = lambda *args, **kwargs: _Completed(
            returncode=0,
            stdout="numpy 1.26.4\n",
            stderr="",
        )
        workarounds._check_numpy_stack_in_current_process = lambda: {
            "ok": True,
            "stdout": "numpy 1.26.4",
            "error": None,
        }
        workarounds._loaded_numpy_stack_modules = lambda: []

        status = workarounds.ensure_numpy_stack_healthy(python_executable="python", repair=False)

        assert status["healthy"] is True
        assert status["current_process_ok"] is True
    finally:
        workarounds.subprocess.run = original_run
        workarounds._check_numpy_stack_in_current_process = original_current_check
        workarounds._loaded_numpy_stack_modules = original_loaded_modules


def test_numpy_stack_check_blocks_poisoned_current_process():
    original_run = workarounds.subprocess.run
    original_current_check = workarounds._check_numpy_stack_in_current_process
    original_loaded_modules = workarounds._loaded_numpy_stack_modules
    try:
        workarounds.subprocess.run = lambda *args, **kwargs: _Completed(
            returncode=0,
            stdout="numpy 1.26.4\n",
            stderr="",
        )
        workarounds._check_numpy_stack_in_current_process = lambda: {
            "ok": False,
            "stdout": None,
            "error": "ValueError('numpy.dtype size changed')",
        }
        workarounds._loaded_numpy_stack_modules = lambda: ["numpy", "numpy._core"]

        try:
            workarounds.ensure_numpy_stack_healthy(python_executable="python", repair=False)
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected RuntimeError for poisoned current process")

        assert "this notebook kernel" in message
        assert "numpy.dtype size changed" in message
    finally:
        workarounds.subprocess.run = original_run
        workarounds._check_numpy_stack_in_current_process = original_current_check
        workarounds._loaded_numpy_stack_modules = original_loaded_modules
