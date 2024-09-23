from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    expected_dir = Path(__file__).resolve().parent / "src"

    current_dir = Path.cwd()

    if not current_dir == expected_dir:
        pytest.exit(
            "Pytest must be run from the project root directory",
            returncode=1,
        )
