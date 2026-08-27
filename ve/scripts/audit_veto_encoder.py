#!/usr/bin/env python3
"""Score an independent speaker encoder on the exact stream used by accepted gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from audio_io import cosine_sim, load_audio
from paths import normalize_presence_label
from presence_encoder import create_presence_encoder


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def rejected(row: dict[str, Any]) -> bool:
    if "reject_decision" in row:
        return bool(row["reject_decision"])
    return str(row.get("decision") or "").startswith("reject")


def exported_path(row: dict[str, Any], condition: str, source: str) -> Path | None:
    for item in (row.get("exported") or {}).get(condition, []):
        if str(item.get("source_stream")) == source and item.get("path"):
            return Path(str(item["path"]))
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit a veto encoder on frozen gate accepts")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--ranked-results", type=Path, required=True)
    p.add_argument("--condition", choices=("raw", "se48k"), default="se48k")
    p.add_argument("--backend", default="campplus")
    p.add_argument("--campplus-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    samples = {str(r["uid"]): r for r in load_jsonl(args.samples)}
    decisions = {str(r["uid"]): r for r in load_jsonl(args.decisions)}
    ranked = {str(r["uid"]): r for r in load_jsonl(args.ranked_results)}
    target_uids = sorted(uid for uid, row in decisions.items() if not rejected(row))
    contract = {
        "backend": args.backend,
        "condition": args.condition,
        "decisions_sha256": hashlib.sha256(args.decisions.read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "ranked_sha256": hashlib.sha256(args.ranked_results.read_bytes()).hexdigest(),
        "enroll_vad": False,
    }
    config_sha = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    done: dict[str, dict[str, Any]] = {}
    if args.resume and args.out.is_file():
        done = {
            str(r["uid"]): r for r in load_jsonl(args.out)
            if r.get("status") == "ok" and r.get("config_sha256") == config_sha
        }
    encoder = create_presence_encoder(
        args.backend, campplus_dir=args.campplus_dir, device=args.device
    )
    output: list[dict[str, Any]] = []
    for index, uid in enumerate(target_uids, 1):
        if uid in done:
            output.append(done[uid])
            continue
        decision = decisions[uid]
        sample = samples.get(uid)
        ranked_row = ranked.get(uid)
        rec: dict[str, Any] = {
            "uid": uid,
            "label": decision.get("label"),
            "split": decision.get("split"),
            "lang": decision.get("lang"),
            "presence_score": decision.get("presence_score"),
            "presence_thr": decision.get("presence_thr"),
            "backend": getattr(encoder, "name", args.backend),
            "condition": args.condition,
            "config_sha256": config_sha,
        }
        t0 = time.perf_counter()
        try:
            if sample is None or ranked_row is None:
                raise KeyError("missing sample or ranked row")
            source = str(decision.get("best_stream") or "")
            target = exported_path(ranked_row, args.condition, source)
            if target is None or not target.is_file():
                raise FileNotFoundError(f"missing exported {args.condition}/{source}")
            enroll_wav, enroll_sr = load_audio(sample["enroll_wav"])
            target_wav, target_sr = load_audio(target)
            enroll_emb = encoder.embed(enroll_wav, enroll_sr)
            target_emb = encoder.embed(target_wav, target_sr)
            score = cosine_sim(enroll_emb, target_emb)
            rec.update({
                "status": "ok", "source_stream": source,
                "target_wav": str(target), "veto_score": round(float(score), 6),
            })
        except Exception as exc:
            rec.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        rec["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        output.append(rec)
        print(f"\r[veto-audit] {index}/{len(target_uids)} uid={uid}".ljust(100), end="\n" if index == len(target_uids) else "", flush=True)
    output.sort(key=lambda r: str(r["uid"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output), encoding="utf-8"
    )
    errors = [r for r in output if r.get("status") != "ok"]
    labels = {
        normalize_presence_label(r.get("label"), split=r.get("split"))
        for r in output if r.get("status") == "ok"
    }
    print(f"[VETO_AUDIT] rows={len(output)} errors={len(errors)} labels={sorted(labels)} -> {args.out}")
    if args.strict and (len(output) != len(target_uids) or errors):
        raise SystemExit(f"veto audit coverage failed errors={len(errors)} first={[r['uid'] for r in errors[:10]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
