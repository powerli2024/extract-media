from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from apply_veto_rescue_overlay import apply_combined  # noqa: E402


def test_rescue_and_veto_are_disjoint_and_veto_wins_its_branch() -> None:
    rows = [
        {"uid": "p", "label": "present", "decision": "reject", "reject_decision": True,
         "reject_reason": "speaker_absent", "pvad": {"applied": True, "decision_eligible": True, "score_aug": .9}},
        {"uid": "n", "label": "absent", "decision": "accept", "reject_decision": False,
         "presence_score": .38, "presence_thr": .34, "pvad": {"applied": True, "decision_eligible": True, "score_aug": .99}},
    ]
    audits = {"n": {"uid": "n", "status": "ok", "presence_score": .38,
                       "presence_thr": .34, "veto_score": .10, "backend": "campplus_zh"}}
    got, summary = apply_combined(
        rows, audits, pvad_threshold=.81, veto_gray=.05, veto_margin=.22,
        config_sha256="cfg", strict=True,
    )
    assert got[0]["combined_action"] == "pvad_rescue" and not got[0]["reject_decision"]
    assert got[1]["combined_action"] == "camp_veto" and got[1]["reject_decision"]
    assert summary["actions"] == {"pvad_rescue": 1, "camp_veto": 1}
