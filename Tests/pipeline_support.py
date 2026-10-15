"""Require current experiment artifacts only for result-dependent tests.

Old result files are not rewritten or treated as new-method evidence.
Unit tests do not use this guard; integration tests resume after the
corresponding pipeline stages have actually been rerun by the user.
"""
import hashlib
import json
from pathlib import Path
from unittest import SkipTest

ROOT=Path(__file__).resolve().parents[1]


def require_current_results(last_stage):
    for stage in range(4,last_stage+1):
        source=next((ROOT/"Sources").glob(f"{stage:02d}_*.py"))
        root=ROOT/"Results"/source.stem
        path=root/("design.json" if stage==9 else "run_manifest.json")
        if not path.exists():
            raise SkipTest(f"Stage {stage:02d} current-method results have not been generated")
        saved=json.loads(path.read_text(encoding="utf8"))
        digest=(saved.get("training_source_sha256") if stage==4 else
                saved.get("source_sha256") if stage==9 else
                saved.get("contract",{}).get("source_sha256"))
        if digest!=hashlib.sha256(source.read_bytes()).hexdigest():
            raise SkipTest(f"Stage {stage:02d} results belong to another source; rerun required")
