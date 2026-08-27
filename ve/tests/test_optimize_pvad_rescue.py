from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from optimize_pvad_rescue import (  # noqa: E402
    evaluate,
    main,
    optimize_thresholds,
    rescue_score,
    stratified_group_split,
)


def row(uid: str, split: str, lang: str, *, rejected: bool, score: float, eligible: bool = True):
    return {
        "uid": uid,
        "split": split,
        "label": "present" if split == "pos" else "absent",
        "lang": lang,
        "enroll_wav": f"{uid}.wav",
        "decision": "reject" if rejected else "accept_no_tse",
        "reject_decision": rejected,
        "cmd_text": "打开" if split == "pos" else "",
        "pvad": {
            "applied": eligible,
            "decision_eligible": eligible,
            "score_aug": score,
            "reason": "trusted_keyframe" if eligible else "untrusted_keyframe",
            "score_aggregation": "top2_mean_approved_streams",
            "config_sha256": "cfg",
        },
    }


def test_rescue_never_changes_static_accept_or_ineligible_reject() -> None:
    assert rescue_score(row("a", "pos", "zh", rejected=False, score=.9)) is None
    assert rescue_score(row("b", "pos", "zh", rejected=True, score=.9, eligible=False)) is None


def test_optimizer_uses_real_asr_gain_and_penalizes_false_rescue() -> None:
    rows = [
        row("p_good", "pos", "zh", rejected=True, score=.8),
        row("p_bad", "pos", "zh", rejected=True, score=.5),
        row("n", "neg", "zh", rejected=True, score=.6),
        row("p_static", "pos", "en", rejected=False, score=.1),
        row("n_static", "neg", "en", rejected=True, score=.1, eligible=False),
    ]
    asr = {
        "p_good": {"uid": "p_good", "status": "ok", "n": 10, "edit_distance": 0},
        "p_bad": {"uid": "p_bad", "status": "ok", "n": 10, "edit_distance": 10},
        "p_static": {"uid": "p_static", "status": "ok", "n": 10, "edit_distance": 0},
    }
    thresholds = optimize_thresholds(rows, asr, "global")
    assert .6 < thresholds["default"] <= .8
    base = evaluate(rows, asr, None)
    candidate = evaluate(rows, asr, thresholds)
    assert candidate["contest_score"] > base["contest_score"]
    assert candidate["n_rescue_pos"] == 1
    assert candidate["n_rescue_neg"] == 0


def test_group_split_never_leaks_enrollment_group() -> None:
    rows = []
    for group in range(8):
        for j in range(2):
            r = row(f"p{group}_{j}", "pos", "zh", rejected=True, score=.8)
            r["enroll_wav"] = f"group{group}.wav"
            rows.append(r)
    train, test = stratified_group_split(rows, .25, 7, "enroll_wav")
    train_groups = {x["enroll_wav"] for x in train}
    test_groups = {x["enroll_wav"] for x in test}
    assert train_groups
    assert test_groups
    assert train_groups.isdisjoint(test_groups)


def test_main_writes_frozen_threshold_only_for_stable_gain(tmp_path: Path, monkeypatch) -> None:
    decisions = []
    asr = []
    for lang in ("zh", "en"):
        for i in range(10):
            uid = f"p_{lang}_{i}"
            decisions.append(row(uid, "pos", lang, rejected=True, score=.8))
            asr.append({
                "uid": uid, "decision": "accept", "cmd_text": "打开",
                "status": "ok", "n": 2, "edit_distance": 0,
            })
            decisions.append(row(f"n_{lang}_{i}", "neg", lang, rejected=True, score=.2))
    dec_path = tmp_path / "dec.jsonl"
    asr_path = tmp_path / "asr.jsonl"
    out = tmp_path / "out"
    dec_path.write_text("".join(json.dumps(x) + "\n" for x in decisions), encoding="utf-8")
    asr_path.write_text("".join(json.dumps(x) + "\n" for x in asr), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "optimize-pvad", "--decisions", str(dec_path), "--asr-all-pos", str(asr_path),
        "--out-dir", str(out), "--seeds", "10", "--holdout-frac", ".3", "--strict",
    ])
    assert main() == 0
    frozen = json.loads((out / "frozen_threshold.json").read_text(encoding="utf-8"))
    assert frozen["thr_mode"] in {"global", "lang_split"}
    assert set(frozen["thr_by_lang"]) == {"zh", "en", "default"}
    assert frozen["paired_delta"]["p05"] > 0
