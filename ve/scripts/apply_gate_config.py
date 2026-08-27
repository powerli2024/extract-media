#!/usr/bin/env python3
"""Apply a recommended gate config to cached similarity rows without labels at runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from optimize_gate_for_score import stream_score


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def stream_source(
    row: dict[str, Any],
    policy: str,
    *,
    rescue_high_margin: float,
    rescue_floor_margin: float,
    rescue_dominance: float,
) -> tuple[str, float, bool]:
    """Return the stream metadata corresponding to ``stream_score``."""
    sims = row.get("sim_streams") or {}
    mix = float(sims.get("mix", row.get("sim_enroll_mix", 0.0)))
    sep = sorted(
        (
            (str(k), float(v))
            for k, v in sims.items()
            if k not in {"mix", "mix_window", "peak"}
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if policy == "mix":
        return "mix", mix, False
    if policy == "max":
        return max([("mix", mix), *sep], key=lambda pair: pair[1]) + (False,)
    if policy != "strict_rescue":
        raise ValueError(f"unknown policy={policy}")
    if len(sep) >= 2 and sep[0][1] - sep[1][1] >= rescue_dominance:
        rescue_cap = min(sep[0][1] - rescue_high_margin, mix + rescue_floor_margin)
        if rescue_cap > mix:
            return sep[0][0], rescue_cap, True
    return "mix", mix, False


def apply_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = load_thresholds_from_data(config)
    policy = str(config.get("stream_policy") or "max")
    margins = (
        float(config.get("rescue_high_margin", .08)),
        float(config.get("rescue_floor_margin", .10)),
        float(config.get("rescue_dominance", .05)),
    )
    out = []
    for original in rows:
        row = dict(original)
        lang = str(row.get("lang") or "zh")
        value = stream_score(
            row, policy, rescue_high_margin=margins[0],
            rescue_floor_margin=margins[1], rescue_dominance=margins[2],
        )
        best_stream, source_value, rescue_eligible = stream_source(
            row,
            policy,
            rescue_high_margin=margins[0],
            rescue_floor_margin=margins[1],
            rescue_dominance=margins[2],
        )
        if abs(value - source_value) > 1e-12:
            raise RuntimeError(
                f"stream metadata mismatch uid={row.get('uid')} "
                f"score={value} source_score={source_value}"
            )
        threshold = float(thresholds.get(lang, thresholds["default"]))
        reject = value < threshold
        sims = row.get("sim_streams") or {}
        row.update({
            "presence_score": value,
            "presence_score_raw": value,
            "presence_thr": threshold,
            "sim_enroll_mix": float(sims.get("mix", row.get("sim_enroll_mix", 0.0))),
            "best_stream": best_stream,
            "score_norm": "raw",
            "stream_policy": policy, "decision": "reject" if reject else "accept",
            "reject_decision": reject,
            "reject_reason": "speaker_absent" if reject else "",
            "rescue_eligible": rescue_eligible,
            "gate_overlay_source": "recommended_thr.json",
        })
        for stale_key in (
            "znorm_mu", "znorm_sigma", "znorm_mu_test", "znorm_sigma_test",
            "z_enroll", "z_test",
        ):
            row.pop(stale_key, None)
        out.append(row)
    return out


def load_thresholds_from_data(data: dict[str, Any]) -> dict[str, float]:
    raw = data.get("thr_by_lang") or {"default": data.get("presence_thr")}
    result = {str(k): float(v) for k, v in raw.items() if v is not None}
    if "default" not in result:
        result["default"] = result.get("zh", next(iter(result.values())))
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="将门控配置应用到已有 sim_streams")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--thr-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    rows = load_jsonl(args.decisions)
    config = json.loads(args.thr_file.read_text(encoding="utf-8"))
    output = apply_rows(rows, config)
    missing = [str(r.get("uid")) for r in rows if not (r.get("sim_streams") or {})]
    if args.strict and missing:
        raise SystemExit(f"missing sim_streams: n={len(missing)} first={missing[:5]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output), encoding="utf-8"
    )
    print(f"[OK] applied gate rows={len(output)} policy={config.get('stream_policy')} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
