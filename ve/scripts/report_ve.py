#!/usr/bin/env python3
"""VE 运行报告与分析。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from paths import normalize_presence_label


def _pct(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def _row_label(row: dict[str, Any]) -> str | None:
    try:
        return normalize_presence_label(row.get("label"), split=row.get("split"))
    except KeyError:
        return None


def contest_score(rr: float, cer: float) -> float:
    """0.5*RR + 0.5*(1-CER)。误拒正样本时该条 CER=1。"""
    return round(0.5 * float(rr) + 0.5 * (1.0 - float(cer)), 6)


def _accepted(row: dict[str, Any]) -> bool:
    """Whether the gate accepted a sample, with or without TSE execution."""
    return row.get("decision") in ("accept", "accept_no_tse")


def _pos_cer(row: dict[str, Any]) -> float | None:
    """正样本 CER：误拒/失败→1；接受必须有 asr_cer/cer，否则 None（未跑 ASR）。"""
    if row.get("decision") == "reject":
        return 1.0
    if row.get("decision") in ("extract_error", "pipeline_error"):
        return 1.0
    for k in ("asr_cer", "cer", "oracle_cer"):
        if row.get(k) is not None:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                pass
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[str(r.get("split") or "?")].append(r)

    out: dict[str, Any] = {"splits": {}, "overall": {}}
    decisions = Counter(r.get("decision") for r in rows)
    out["overall"]["decisions"] = dict(decisions)
    out["overall"]["n"] = len(rows)

    accept = [r for r in rows if _accepted(r)]
    reject = [r for r in rows if r.get("decision") == "reject"]
    out["overall"]["n_accept"] = len(accept)
    out["overall"]["n_reject"] = len(reject)
    if rows:
        scores = [float(r["presence_score"]) for r in rows if r.get("presence_score") is not None]
        out["overall"]["mean_presence_score"] = round(sum(scores) / len(scores), 4) if scores else None
        ms = [float(r["elapsed_ms"]) for r in rows if r.get("elapsed_ms") is not None]
        if ms:
            ms_sorted = sorted(ms)
            out["overall"]["latency_ms"] = {
                "mean": round(sum(ms) / len(ms), 1),
                "p50": ms_sorted[len(ms_sorted) // 2],
                "p95": ms_sorted[min(len(ms_sorted) - 1, int(0.95 * (len(ms_sorted) - 1)))],
            }

    for split, rs in by_split.items():
        n = len(rs)
        n_acc = sum(1 for r in rs if _accepted(r))
        n_rej = sum(1 for r in rs if r.get("decision") == "reject")
        n_err = sum(1 for r in rs if r.get("decision") in ("extract_error", "pipeline_error"))
        present_rs = [r for r in rs if _row_label(r) == "present"]
        absent_rs = [r for r in rs if _row_label(r) == "absent"]
        if present_rs and not absent_rs:
            label = "present"
        elif absent_rs and not present_rs:
            label = "absent"
        elif present_rs and absent_rs:
            label = "mixed"
        else:
            label = _row_label(rs[0]) if rs else None
        block = {
            "n": n,
            "label": label,
            "n_present": len(present_rs),
            "n_absent": len(absent_rs),
            "accept_rate": _pct(n_acc, n),
            "reject_rate": _pct(n_rej, n),
            "error_rate": _pct(n_err, n),
            "mean_presence_score": round(
                sum(float(r["presence_score"]) for r in rs if r.get("presence_score") is not None)
                / max(1, sum(1 for r in rs if r.get("presence_score") is not None)),
                4,
            ),
        }
        if present_rs:
            n_p = len(present_rs)
            n_p_rej = sum(1 for r in present_rs if r.get("decision") == "reject")
            block["n_present_false_reject"] = n_p_rej
            block["frr"] = _pct(n_p_rej, n_p)
            cers = [_pos_cer(r) for r in present_rs]
            known = [c for c in cers if c is not None]
            missing = sum(1 for c in cers if c is None)
            if missing:
                block["cer"] = None
                block["cer_note"] = (
                    f"缺少 asr_cer 的样本 {missing}/{n_p}；请跑 ./run_asr_cer.sh 后再汇总"
                )
            else:
                block["cer"] = round(sum(known) / n_p, 4) if n_p else 0.0
        if absent_rs:
            n_a = len(absent_rs)
            n_a_acc = sum(1 for r in absent_rs if _accepted(r))
            n_a_rej = sum(1 for r in absent_rs if r.get("decision") == "reject")
            block["n_absent_correct_reject"] = n_a_rej
            block["n_absent_false_accept"] = n_a_acc
            block["far"] = _pct(n_a_acc, n_a)
            block["rr"] = _pct(n_a_rej, n_a)
        out["splits"][split] = block

    # 竞赛总分
    pos = [r for r in rows if _row_label(r) == "present"]
    neg = [r for r in rows if _row_label(r) == "absent"]
    rr = _pct(sum(1 for r in neg if r.get("decision") == "reject"), len(neg)) if neg else 0.0
    pos_cers = [_pos_cer(r) for r in pos]
    missing_asr = sum(1 for c in pos_cers if c is None)
    out["overall"]["rr"] = rr
    n_pos_false_reject = sum(1 for r in pos if r.get("decision") == "reject")
    out["overall"]["n_pos"] = len(pos)
    out["overall"]["n_neg"] = len(neg)
    out["overall"]["n_pos_false_reject"] = n_pos_false_reject
    out["overall"]["frr"] = _pct(n_pos_false_reject, len(pos)) if pos else 0.0
    # 只用 Presence 决策可得到的代理分，用于门控调试；不是最终竞赛分。
    out["overall"]["presence_proxy_score"] = contest_score(
        rr, out["overall"]["frr"]
    )
    if missing_asr:
        out["overall"]["cer"] = None
        out["overall"]["cer_pending_asr"] = missing_asr
        out["overall"]["metric"] = (
            "最终 CER/竞赛分需先跑 ./run_asr_cer.sh + final_evaluate.py"
        )
    else:
        cer_macro = (sum(pos_cers) / len(pos)) if pos else 0.0  # type: ignore[arg-type]
        out["overall"]["cer_macro_diagnostic"] = round(float(cer_macro), 4)
        out["overall"]["metric"] = (
            "CER 宏平均仅诊断；最终字符加权 CER/竞赛分见 final_evaluate.py"
        )

    bad_reasons = [
        r["uid"]
        for r in reject
        if r.get("reject_reason") and r.get("reject_reason") != "speaker_absent"
    ]
    out["audit"] = {
        "reject_reasons": dict(Counter(r.get("reject_reason") for r in reject)),
        "non_absent_reject_uids": bad_reasons[:50],
        "n_non_absent_reject": len(bad_reasons),
        "n_rescue_eligible": sum(bool(r.get("rescue_eligible")) for r in rows),
        "n_rescue_accept": sum(
            bool(r.get("rescue_eligible")) and r.get("decision") == "accept"
            for r in rows
        ),
    }
    return out


def write_run_reports(
    reports_dir: Path,
    rows: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    if meta:
        summary["meta"] = meta

    (reports_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# VE 运行报告（Presence-gated TSE）",
        "",
        "## 拒识口径",
        "",
        "- **仅** `speaker_absent`（presence_score < thr）",
        "- 不使用能量 / ASR / TSE 后验相似度拒识",
        "",
        "## Meta",
        "",
        "```json",
        json.dumps(meta or {}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Presence 预览（非最终排名）",
        "",
        f"- n={summary['overall']['n']} accept={summary['overall']['n_accept']} "
        f"reject={summary['overall']['n_reject']}",
        f"- 负样本 RR={summary['overall'].get('rr')}；正样本 FRR="
        f"{summary['overall'].get('frr')}（误拒 {summary['overall'].get('n_pos_false_reject')}/"
        f"{summary['overall'].get('n_pos')}）",
        f"- Presence proxy = 0.5*RR + 0.5*(1-FRR) = "
        f"**{summary['overall'].get('presence_proxy_score')}**（只用于门控诊断）",
        "- 最终 **CER** 与 **contest_score** 必须读取 `reports/final_eval/summary.json`。",
        f"- decisions: `{summary['overall'].get('decisions')}`",
        f"- latency: `{summary['overall'].get('latency_ms')}`",
        "",
        "## Per split",
        "",
    ]
    for split, b in summary["splits"].items():
        lines.append(f"### {split} (label={b.get('label')})")
        lines.append("")
        lines.append(f"- n={b['n']} accept_rate={b['accept_rate']} reject_rate={b['reject_rate']}")
        if "frr" in b:
            lines.append(
                f"- FRR(present mis-reject)={b['frr']} "
                f"({b.get('n_present_false_reject')}/{b.get('n_present')})"
            )
        if "cer" in b:
            lines.append(f"- CER(pos; mis-reject=1)={b['cer']}")
        if "far" in b:
            lines.append(f"- FAR(absent mis-accept)={b['far']}")
        if "rr" in b:
            lines.append(
                f"- RR(neg correct-reject)={b['rr']} "
                f"({b.get('n_absent_correct_reject')}/{b.get('n_absent')})"
            )
        lines.append(f"- mean_presence_score={b['mean_presence_score']}")
        lines.append("")

    lines += [
        "## Audit",
        "",
        f"- reject_reasons: `{summary['audit']['reject_reasons']}`",
        f"- n_non_absent_reject: {summary['audit']['n_non_absent_reject']} (应为 0)",
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    # 失败清单
    fails = [
        r
        for r in rows
        if r.get("decision") in ("extract_error", "pipeline_error")
        or (_row_label(r) == "present" and r.get("decision") == "reject")
        or (_row_label(r) == "absent" and _accepted(r))
    ]
    with (reports_dir / "failures.jsonl").open("w", encoding="utf-8") as f:
        for r in fails:
            f.write(
                json.dumps(
                    {
                        "uid": r.get("uid"),
                        "split": r.get("split"),
                        "label": r.get("label"),
                        "decision": r.get("decision"),
                        "presence_score": r.get("presence_score"),
                        "reject_reason": r.get("reject_reason"),
                        "error": r.get("error") or r.get("extract_error"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    analysis = {
        "n_failures_listed": len(fails),
        "present_rejected": sum(
            1 for r in rows if _row_label(r) == "present" and r.get("decision") == "reject"
        ),
        "absent_accepted": sum(
            1 for r in rows if _row_label(r) == "absent" and _accepted(r)
        ),
        "extract_errors": sum(1 for r in rows if r.get("decision") == "extract_error"),
    }
    (reports_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
