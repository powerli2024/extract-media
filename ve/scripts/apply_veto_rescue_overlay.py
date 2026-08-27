#!/usr/bin/env python3
"""Apply frozen PVAD rescue and independent-encoder veto to gate decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from paths import normalize_presence_label


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def rejected(row: dict[str, Any]) -> bool:
    if "reject_decision" in row:
        return bool(row["reject_decision"])
    return str(row.get("decision") or "").startswith("reject")


def pvad_rescued(row: dict[str, Any], threshold: float) -> bool:
    pvad = row.get("pvad") or {}
    return bool(
        rejected(row)
        and pvad.get("applied")
        and pvad.get("decision_eligible")
        and pvad.get("score_aug") is not None
        and float(pvad["score_aug"]) >= threshold
    )


def vetoed(audit: dict[str, Any], gray: float, margin: float) -> bool:
    score = float(audit["presence_score"])
    threshold = float(audit["presence_thr"])
    veto_score = float(audit["veto_score"])
    return 0.0 <= score - threshold <= gray and veto_score < score - margin


def apply_combined(
    rows: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    *,
    pvad_threshold: float,
    veto_gray: float,
    veto_margin: float,
    config_sha256: str,
    strict: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    expected_audits = {str(r["uid"]) for r in rows if not rejected(r)}
    if strict and set(audits) != expected_audits:
        raise ValueError(
            f"veto audit UID mismatch missing={sorted(expected_audits-set(audits))[:5]} "
            f"extra={sorted(set(audits)-expected_audits)[:5]}"
        )
    for original in rows:
        row = dict(original)
        uid = str(row["uid"])
        action = "unchanged"
        final_reject = rejected(row)
        if final_reject:
            if pvad_rescued(row, pvad_threshold):
                final_reject = False
                action = "pvad_rescue"
        else:
            audit = audits.get(uid)
            if audit is None:
                if strict:
                    raise ValueError(f"missing veto audit uid={uid}")
            else:
                if audit.get("status") != "ok":
                    raise ValueError(f"bad veto audit uid={uid} status={audit.get('status')}")
                if strict and (
                    abs(float(audit["presence_score"]) - float(row["presence_score"])) > 1e-6
                    or abs(float(audit["presence_thr"]) - float(row["presence_thr"])) > 1e-6
                ):
                    raise ValueError(f"veto audit score contract mismatch uid={uid}")
                row["veto_score"] = float(audit["veto_score"])
                row["veto_backend"] = str(audit.get("backend") or "campplus")
                if vetoed(audit, veto_gray, veto_margin):
                    final_reject = True
                    action = "camp_veto"
        row.update({
            "decision": "reject" if final_reject else "accept",
            "reject_decision": final_reject,
            "reject_reason": "camp_veto" if action == "camp_veto" else (
                str(original.get("reject_reason") or "speaker_absent")
                if final_reject else ""
            ),
            "combined_action": action,
            "combined_config_sha256": config_sha256,
        })
        actions[action] += 1
        label = normalize_presence_label(row.get("label"), split=row.get("split"))
        if final_reject:
            rejects[label] += 1
        output.append(row)
    summary = {
        "n_rows": len(output),
        "actions": dict(actions),
        "rejects": dict(rejects),
        "pvad_threshold": pvad_threshold,
        "veto_gray": veto_gray,
        "veto_margin": veto_margin,
        "config_sha256": config_sha256,
    }
    return output, summary


def main() -> int:
    p = argparse.ArgumentParser(description="Apply frozen PVAD rescue + veto overlay")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--veto-audit", type=Path, required=True)
    p.add_argument("--pvad-threshold", type=float, required=True)
    p.add_argument("--veto-gray", type=float, required=True)
    p.add_argument("--veto-margin", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    config = {
        "pvad_threshold": args.pvad_threshold,
        "veto_gray": args.veto_gray,
        "veto_margin": args.veto_margin,
        "precedence": "static_reject->pvad_rescue;static_accept->camp_veto",
        "decisions_sha256": hashlib.sha256(args.decisions.read_bytes()).hexdigest(),
        "veto_audit_sha256": hashlib.sha256(args.veto_audit.read_bytes()).hexdigest(),
    }
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    rows = load_jsonl(args.decisions)
    audit_rows = load_jsonl(args.veto_audit)
    audits = {str(r["uid"]): r for r in audit_rows}
    if args.strict and len(audits) != len(audit_rows):
        raise SystemExit("duplicate veto audit UIDs")
    output, summary = apply_combined(
        rows, audits, pvad_threshold=args.pvad_threshold,
        veto_gray=args.veto_gray, veto_margin=args.veto_margin,
        config_sha256=config_sha, strict=args.strict,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output), encoding="utf-8"
    )
    summary["config"] = config
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] combined rows={len(output)} actions={summary['actions']} rejects={summary['rejects']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
