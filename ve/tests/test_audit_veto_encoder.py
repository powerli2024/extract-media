from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_veto_encoder import exported_path, rejected  # noqa: E402


def test_rejected_supports_overlay_contract() -> None:
    assert rejected({"reject_decision": True})
    assert not rejected({"decision": "accept"})


def test_exported_path_selects_exact_condition_and_source() -> None:
    row = {
        "exported": {
            "raw": [{"source_stream": "mix", "path": "/raw.wav"}],
            "se48k": [
                {"source_stream": "d1_spk1", "path": "/s1.wav"},
                {"source_stream": "d1_spk2", "path": "/s2.wav"},
            ],
        }
    }
    assert exported_path(row, "se48k", "d1_spk2") == Path("/s2.wav")
    assert exported_path(row, "raw", "d1_spk2") is None
