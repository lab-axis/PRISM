"""Regression: PRISM Stages 05–08 do not stop on elapsed time."""
from pathlib import Path

import pytest


SOURCES = Path(__file__).resolve().parents[1] / "Sources"


@pytest.mark.parametrize("stage", ("05", "06", "07", "08"))
def test_training_has_no_wall_clock_budget(stage):
    files = list(SOURCES.glob(f"{stage}_*.py"))
    assert len(files) == 1
    code = files[0].read_text(encoding="utf-8")
    for forbidden in (
        "--max-minutes", "max_minutes", "budget_minutes",
        "PARTIAL_TIME_BUDGET", "PAUSED_RESUMABLE"):
        assert forbidden not in code
