from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from apply_gate_config import apply_rows  # noqa: E402


def test_apply_gate_uses_language_threshold_and_preserves_rows() -> None:
    rows = [
        {
            "uid": "z", "lang": "zh",
            "sim_streams": {"mix": .25, "d1_spk1": .4},
            "presence_score_raw": .01, "best_stream": "stale",
            "score_norm": "asnorm", "znorm_mu": .1,
        },
        {"uid": "e", "lang": "en", "sim_streams": {"mix": .25, "d1_spk1": .2}},
    ]
    config = {
        "stream_policy": "max",
        "thr_by_lang": {"zh": .2, "en": .3, "default": .2},
    }
    got = apply_rows(rows, config)
    assert got[0]["decision"] == "accept"
    assert got[1]["decision"] == "reject"
    assert got[0]["presence_score"] == .4
    assert got[0]["presence_score_raw"] == .4
    assert got[0]["sim_enroll_mix"] == .25
    assert got[0]["best_stream"] == "d1_spk1"
    assert got[0]["score_norm"] == "raw"
    assert got[0]["rescue_eligible"] is False
    assert "znorm_mu" not in got[0]
    assert rows[0].get("decision") is None


def test_apply_gate_recomputes_strict_rescue_source() -> None:
    rows = [{
        "uid": "p", "lang": "zh",
        "sim_streams": {"mix": .20, "d1_spk1": .50, "d1_spk2": .30},
    }]
    config = {
        "stream_policy": "strict_rescue",
        "thr_by_lang": {"zh": .25, "default": .25},
        "rescue_high_margin": .08,
        "rescue_floor_margin": .10,
        "rescue_dominance": .05,
    }
    got = apply_rows(rows, config)[0]
    assert abs(got["presence_score"] - .30) < 1e-12
    assert got["presence_score_raw"] == got["presence_score"]
    assert got["best_stream"] == "d1_spk1"
    assert got["rescue_eligible"] is True
