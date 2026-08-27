from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_extract import load_pvad_decision_thresholds  # noqa: E402


def test_load_language_pvad_threshold_contract(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({
        "thr_mode": "lang_split",
        "thr_by_lang": {"zh": .42, "en": .51, "default": .42},
        "score_aggregation": "top2_mean_approved_streams",
    }), encoding="utf-8")
    thresholds, contract = load_pvad_decision_thresholds(path, None)
    assert thresholds == {"zh": .42, "en": .51, "default": .42}
    assert contract is not None
    assert contract["thr_mode"] == "lang_split"


def test_scalar_and_file_are_mutually_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    path.write_text('{"pvad_decision_thr": 0.4}', encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_pvad_decision_thresholds(path, .4)


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        load_pvad_decision_thresholds(None, 1.2)
