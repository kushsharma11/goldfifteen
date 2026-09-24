"""Run the offline CLI pipeline in pytest's isolated temporary workspace.

No production data is generated, no network request is made, and fixture
profit/loss must not be interpreted as actual market performance.
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_cli.py::test_full_offline_research_workflow",
            ],
            cwd=Path(__file__).resolve().parents[1],
        )
    )
