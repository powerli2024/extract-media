#!/usr/bin/env python3
"""Optimize ASE-PVAD rescue thresholds with real all-positive ASR costs.

The frozen Presence gate remains authoritative.  A threshold can only rescue a
row that the frozen gate rejected and whose audit record says the ASE evidence
was both trusted and decision-eligible.  Static accepts are never revoked.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from paths import normalize_presence_label


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def is_rejected(row: dict[str, Any]) -> bool:
    if "reject" in row:
        return bool(row["reject"])
    decision = str(row.get("final_decision") or row.get("decision") or "")
    return decision.startswith("reject") or bool(row.get("reject_decision"))


def rescue_score(row: dict[str, Any]) -> float | None:
    """Return the calibrated score only for rows legally eligible for rescue."""
    if not is_rejected(row):
        return None
    pvad = row.get("pvad")
    if not isinstance(pvad, dict):
        return None
    if not bool(pvad.get("applied")) or not bool(pvad.get("decision_eligible")):
        return None
    value = pvad.get("score_aug")
    if value is None:
        return None
    score = float(value)
    if not np.isfinite(score) or not -1.0 <= score <= 1.0:
        raise ValueError(f"invalid PVAD score uid={row.get('uid')} score={value}")
    return score


def asr_cost(row: dict[str, Any] | None) -> tuple[int, int]:
    if not row:
        raise KeyError("missing ASR row")
    n = int(row.get("n") or 0)
    if n <= 0:
        raise ValueError(f"invalid ASR ref length uid={row.get('uid')} n={n}")
    if row.get("status") != "ok" or row.get("edit_distance") is None:
        return n, n
    return int(row["edit_distance"]), n


def threshold_for(row: dict[str, Any], thresholds: dict[str, float]) -> float:
    lang = str(row.get("lang") or "zh")
    return float(thresholds.get(lang, thresholds["default"]))


def evaluate(
    rows: list[dict[str, Any]],
    asr: dict[str, dict[str, Any]],
    thresholds: dict[str, float] | None,
) -> dict[str, Any]:
    n_pos = n_neg = n_rej_pos = n_rej_neg = errors = refs = rescues = 0
    rescue_pos = rescue_neg = 0
    for row in rows:
        label = normalize_presence_label(row.get("label"), split=row.get("split"))
        rejected = is_rejected(row)
        score = rescue_score(row)
        rescued = bool(
            rejected and thresholds is not None and score is not None
            and score >= threshold_for(row, thresholds)
        )
        if rescued:
            rejected = False
            rescues += 1
            rescue_pos += int(label == "present")
            rescue_neg += int(label == "absent")
        if label == "absent":
            n_neg += 1
            n_rej_neg += int(rejected)
            continue
        n_pos += 1
        n_rej_pos += int(rejected)
        err, n = asr_cost(asr.get(str(row["uid"])))
        refs += n
        errors += n if rejected else err
    rr = n_rej_neg / max(1, n_neg)
    cer = errors / max(1, refs)
    return {
        "n_pos": n_pos, "n_neg": n_neg,
        "rr": rr, "frr": n_rej_pos / max(1, n_pos),
        "cer_pos_micro": cer, "contest_score": 0.5 * (rr + 1.0 - cer),
        "pos_errors": errors, "pos_ref_chars": refs,
        "n_rescues": rescues, "n_rescue_pos": rescue_pos, "n_rescue_neg": rescue_neg,
    }


def optimize_thresholds(
    rows: list[dict[str, Any]],
    asr: dict[str, dict[str, Any]],
    mode: str,
) -> dict[str, float]:
    if mode not in {"global", "lang_split"}:
        raise ValueError(f"unknown threshold mode: {mode}")
    n_neg = sum(
        normalize_presence_label(r.get("label"), split=r.get("split")) == "absent"
        for r in rows
    )
    total_ref = sum(
        asr_cost(asr.get(str(r["uid"])))[1]
        for r in rows
        if normalize_presence_label(r.get("label"), split=r.get("split")) == "present"
    )
    if not n_neg or not total_ref:
        raise ValueError("optimization requires negatives and positive ASR references")

    groups = [("default", None)] if mode == "global" else [("zh", "zh"), ("en", "en")]
    out: dict[str, float] = {}
    for key, lang in groups:
        events: list[tuple[float, float]] = []
        for row in rows:
            if lang is not None and str(row.get("lang") or "zh") != lang:
                continue
            score = rescue_score(row)
            if score is None:
                continue
            label = normalize_presence_label(row.get("label"), split=row.get("split"))
            if label == "absent":
                delta = -0.5 / n_neg
            else:
                err, n = asr_cost(asr.get(str(row["uid"])))
                delta = 0.5 * (n - err) / total_ref
            events.append((score, delta))
        if not events:
            out[key] = 1.0
            continue
        events.sort(key=lambda x: x[0], reverse=True)
        best_gain = gain = 0.0
        best_thr = min(1.0, events[0][0] + 1e-6)
        i = 0
        while i < len(events):
            score = events[i][0]
            while i < len(events) and events[i][0] == score:
                gain += events[i][1]
                i += 1
            if gain > best_gain + 1e-15 or (
                abs(gain - best_gain) <= 1e-15 and score > best_thr
            ):
                best_gain, best_thr = gain, score
        out[key] = float(best_thr)
    if mode == "global":
        out["zh"] = out["en"] = out["default"]
    else:
        if "zh" not in out:
            out["zh"] = 1.0
        if "en" not in out:
            out["en"] = out["zh"]
        out["default"] = out["zh"]
    return out


def stratified_group_split(
    rows: list[dict[str, Any]], frac: float, seed: int, group_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split atomic enrollment groups inside label/language strata."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row.get(group_field) or row.get("uid"))
        grouped[group].append(row)
    buckets: dict[tuple[str, str], list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for group, members in grouped.items():
        signatures = Counter(
            (
                normalize_presence_label(r.get("label"), split=r.get("split")),
                str(r.get("lang") or "zh"),
            )
            for r in members
        )
        buckets[signatures.most_common(1)[0][0]].append((group, members))
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for groups in buckets.values():
        rng.shuffle(groups)
        n_test = max(1, int(round(frac * len(groups)))) if groups else 0
        for _group, members in groups[:n_test]:
            test.extend(members)
        for _group, members in groups[n_test:]:
            train.extend(members)
    return train, test


def dist(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(x)), "std": float(np.std(x)),
        "p05": float(np.quantile(x, .05)), "p50": float(np.quantile(x, .50)),
        "p95": float(np.quantile(x, .95)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize ASE-PVAD rescue using official score")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--asr-all-pos", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threshold-modes", default="global,lang_split")
    p.add_argument("--holdout-frac", type=float, default=.30)
    p.add_argument("--seeds", type=int, default=500)
    p.add_argument("--group-field", default="enroll_wav")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.decisions.resolve())
    asr_rows = load_jsonl(args.asr_all_pos.resolve())
    dec = {str(r["uid"]): r for r in rows if r.get("uid")}
    asr = {str(r["uid"]): r for r in asr_rows if r.get("uid")}
    pos = {
        uid: r for uid, r in dec.items()
        if normalize_presence_label(r.get("label"), split=r.get("split")) == "present"
    }
    errors: list[str] = []
    if len(dec) != len(rows):
        errors.append(f"duplicate_decisions={len(rows) - len(dec)}")
    if len(asr) != len(asr_rows):
        errors.append(f"duplicate_asr={len(asr_rows) - len(asr)}")
    missing = sorted(set(pos) - set(asr))
    if missing:
        errors.append(f"missing_positive_asr={len(missing)}")
    for uid in sorted(set(pos) & set(asr)):
        try:
            if asr[uid].get("status") != "ok" or asr[uid].get("decision") != "accept":
                raise ValueError("not forced-accept ok")
            if str(asr[uid].get("cmd_text") or "") != str(pos[uid].get("cmd_text") or ""):
                raise ValueError("cmd_text mismatch")
            asr_cost(asr[uid])
        except (KeyError, ValueError) as exc:
            errors.append(f"{uid}:{exc}")
    pvad_rows = [r.get("pvad") for r in rows if isinstance(r.get("pvad"), dict)]
    if len(pvad_rows) != len(rows):
        errors.append(f"missing_pvad_rows={len(rows) - len(pvad_rows)}")
    if any(p.get("reason") == "pvad_ase_exception" for p in pvad_rows):
        errors.append("pvad_exceptions_present")
    aggregations = {str(p.get("score_aggregation")) for p in pvad_rows}
    if aggregations != {"top2_mean_approved_streams"}:
        errors.append(f"unexpected_score_aggregation={sorted(aggregations)}")
    if args.strict and errors:
        raise SystemExit("coverage invalid: " + "; ".join(errors[:20]))

    modes = [x.strip() for x in args.threshold_modes.split(",") if x.strip()]
    baseline_full = evaluate(rows, asr, None)
    report: dict[str, Any] = {
        "metric": "0.5*RR_neg + 0.5*(1-CER_pos_micro); rejected positive has errors=N",
        "contract": "static accepts unchanged; only trusted eligible static rejects may be rescued",
        "coverage": {
            "n_decisions": len(rows), "n_pos": len(pos), "n_asr": len(asr_rows),
            "errors": errors, "n_rescue_eligible": sum(rescue_score(r) is not None for r in rows),
        },
        "baseline_full": baseline_full,
        "full_data_diagnostic": {}, "holdout": {},
    }
    folds: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    for mode in modes:
        thr = optimize_thresholds(rows, asr, mode)
        report["full_data_diagnostic"][mode] = {
            "thresholds": thr, "metrics": evaluate(rows, asr, thr),
            "warning": "in-sample oracle; never deploy this threshold",
        }
    for seed in range(max(1, args.seeds)):
        train, test = stratified_group_split(rows, args.holdout_frac, seed, args.group_field)
        base = evaluate(test, asr, None)
        for mode in modes:
            thr = optimize_thresholds(train, asr, mode)
            met = evaluate(test, asr, thr)
            folds[mode].append({
                **met, "score_delta_vs_baseline": met["contest_score"] - base["contest_score"],
                "thr_zh": thr["zh"], "thr_en": thr["en"], "thr_default": thr["default"],
            })
    for mode, vals in folds.items():
        report["holdout"][mode] = {
            "distributions": {
                key: dist([float(v[key]) for v in vals])
                for key in (
                    "contest_score", "score_delta_vs_baseline", "rr", "cer_pos_micro",
                    "frr", "n_rescues", "n_rescue_pos", "n_rescue_neg",
                    "thr_zh", "thr_en", "thr_default",
                )
            }
        }
    stable = [m for m in modes if report["holdout"][m]["distributions"]["score_delta_vs_baseline"]["p05"] > 0]
    ranked = sorted(
        stable or modes,
        key=lambda m: (
            report["holdout"][m]["distributions"]["score_delta_vs_baseline"]["p05"],
            report["holdout"][m]["distributions"]["contest_score"]["p05"],
        ),
        reverse=True,
    )
    best = ranked[0]
    deployable = best in stable
    thresholds = {
        lang: float(np.median([v[f"thr_{lang}"] for v in folds[best]]))
        for lang in ("zh", "en", "default")
    }
    report["recommendation"] = {
        "deployable": deployable, "threshold_mode": best,
        "thresholds": thresholds if deployable else None,
        "basis": "paired holdout delta p05 > 0" if deployable else "no stable holdout gain",
        "threshold_estimator": "median of repeated training-fold optima",
        "requires_independent_validation": True,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pvad_rescue_optimization.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if deployable:
        config_shas = sorted({str(p.get("config_sha256")) for p in pvad_rows})
        frozen = {
            "pvad_decision_thr": thresholds["default"],
            "thr_mode": best, "thr_by_lang": thresholds,
            "score_aggregation": "top2_mean_approved_streams",
            "config_sha256": config_shas[0] if len(config_shas) == 1 else config_shas,
            "source": "repeated_group_stratified_holdout_official_score",
            "holdout_frac": args.holdout_frac, "seeds": args.seeds,
            "group_field": args.group_field,
            "paired_delta": report["holdout"][best]["distributions"]["score_delta_vs_baseline"],
            "warning": "KWS-specific; validate independently without retuning",
        }
        (args.out_dir / "frozen_threshold.json").write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    lines = [
        "# ASE-PVAD rescue optimization", "",
        f"Deployable: **{'YES' if deployable else 'NO'}**", "",
        "| mode | score p05 | delta mean | delta p05 | RR mean | CER mean | rescue pos | rescue neg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        d = report["holdout"][mode]["distributions"]
        lines.append(
            f"| {mode} | {d['contest_score']['p05']:.6f} | "
            f"{d['score_delta_vs_baseline']['mean']:.6f} | {d['score_delta_vs_baseline']['p05']:.6f} | "
            f"{d['rr']['mean']:.6f} | {d['cer_pos_micro']['mean']:.6f} | "
            f"{d['n_rescue_pos']['mean']:.2f} | {d['n_rescue_neg']['mean']:.2f} |"
        )
    (args.out_dir / "pvad_rescue_optimization.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if deployable else 3


if __name__ == "__main__":
    raise SystemExit(main())
