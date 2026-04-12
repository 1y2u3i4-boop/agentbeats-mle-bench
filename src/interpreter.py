"""Execute Python scripts in isolated subprocesses."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    stdout: str
    exec_time: float
    exc_type: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.exc_type is None


def _classify_error(stderr: str) -> str:
    stderr_lower = stderr.lower()
    for pattern, err_type in [
        ("modulenotfounderror", "ImportError"),
        ("no module named", "ImportError"),
        ("memoryerror", "MemoryError"),
        ("killed", "MemoryError"),
        ("syntaxerror", "SyntaxError"),
        ("keyerror", "KeyError"),
        ("filenotfounderror", "FileNotFoundError"),
        ("valueerror", "ValueError"),
        ("typeerror", "TypeError"),
        ("indexerror", "IndexError"),
        ("zerodivisionerror", "ZeroDivisionError"),
        ("runtimeerror", "RuntimeError"),
        ("permissionerror", "PermissionError"),
    ]:
        if pattern in stderr_lower:
            return err_type
    return "RuntimeError"


class Interpreter:
    def __init__(self, *, workdir: Path, timeout: int = 600):
        self.workdir = workdir
        self.timeout = timeout
        self._script_path: Path | None = None

    def run(self, code: str) -> ExecutionResult:
        script = self.workdir / "_solver_script.py"
        script.write_text(code, encoding="utf-8")
        self._script_path = script

        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start
            combined = result.stdout
            if result.stderr:
                combined += "\n--- STDERR ---\n" + result.stderr

            if result.returncode != 0:
                err_type = _classify_error(result.stderr)
                # Include last 2000 chars of stderr for debugging
                stderr_tail = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
                return ExecutionResult(
                    stdout=combined + "\n" + stderr_tail,
                    exec_time=elapsed,
                    exc_type=err_type,
                )
            return ExecutionResult(stdout=combined, exec_time=elapsed)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return ExecutionResult(
                stdout=f"Script timed out after {self.timeout}s",
                exec_time=elapsed,
                exc_type="TimeoutError",
            )
        except Exception as e:
            elapsed = time.time() - start
            return ExecutionResult(
                stdout=str(e),
                exec_time=elapsed,
                exc_type=type(e).__name__,
            )

    def cleanup(self):
        if self._script_path and self._script_path.exists():
            self._script_path.unlink(missing_ok=True)
