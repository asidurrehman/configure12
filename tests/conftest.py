import os
import sys
from pathlib import Path

import pytest

# Plots are saved to files; never open a window during the tests.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def cf():
    """The CONFIGURE v1.2 function module."""
    import configure12_functions
    return configure12_functions


@pytest.fixture
def fake_citycat(monkeypatch):
    """Replace calls to citycat.exe with the synthetic model in synthetic_case."""
    import subprocess
    import synthetic_case
    monkeypatch.setattr(subprocess, "call", synthetic_case.fake_citycat)
    return synthetic_case.fake_citycat
