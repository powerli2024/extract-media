from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from report_ve import summarize, write_run_reports  # noqa: E402


def test_presence_report_exposes_rr_frr_counts_and_proxy() -> None:
    rows = [
        {"uid": "p0", "split": "pos", "label": "present", "decision": "reject"},
        {"uid": "p1", "split": "pos", "label": "present", "decision": "accept"},
        {"uid": "n0", "split": "neg", "label": "absent", "decision": "reject"},
        {"uid": "n1", "split": "neg", "label": "absent", "decision": "accept"},
    ]
    overall = summarize(rows)["overall"]
    assert overall["rr"] == 0.5
    assert overall["frr"] == 0.5
    assert overall["n_pos_false_reject"] == 1
    assert overall["presence_proxy_score"] == 0.5
    assert "contest_score" not in overall


def test_presence_report_counts_accept_without_tse() -> None:
    rows = [
        {"uid": "p0", "split": "pos", "label": "present", "decision": "accept_no_tse"},
        {"uid": "n0", "split": "neg", "label": "absent", "decision": "accept_no_tse"},
    ]
    report = summarize(rows)
    assert report["overall"]["n_accept"] == 2
    assert report["splits"]["pos"]["accept_rate"] == 1.0
    assert report["splits"]["neg"]["far"] == 1.0


def test_failure_list_includes_accept_without_tse_for_negative(tmp_path: Path) -> None:
    write_run_reports(
        tmp_path,
        [{"uid": "n0", "split": "neg", "label": "absent", "decision": "accept_no_tse"}],
    )
    assert '"uid": "n0"' in (tmp_path / "failures.jsonl").read_text(encoding="utf-8")
    assert '"absent_accepted": 1' in (tmp_path / "analysis.json").read_text(encoding="utf-8")
