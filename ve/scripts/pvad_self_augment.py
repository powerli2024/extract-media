"""P0 ASE-PVAD: stateless, auditable speaker-embedding self augmentation.

This module deliberately has no label/text input.  It selects a trusted CMD
window, forms one embedding anchored to the enrollment embedding, and returns
audit fields; callers decide whether that evidence may change a gate decision.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n <= 1e-12:
        raise ValueError("invalid speaker embedding")
    return x / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_norm(a), _norm(b)))


@dataclass(frozen=True)
class ASEConfig:
    win_sec: float = 1.0
    hop_sec: float = 0.2
    lam: float = 0.10
    seed_abs: float = 0.0
    top2_margin: float = 0.04
    min_consecutive: int = 2
    support_margin: float = 0.0


def windows(wav: np.ndarray, sr: int, cfg: ASEConfig) -> list[tuple[int, int, np.ndarray]]:
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = max(1, int(round(cfg.win_sec * sr)))
    h = max(1, int(round(cfg.hop_sec * sr)))
    if x.size <= n:
        return [(0, x.size, np.pad(x, (0, max(0, n - x.size))))]
    starts = list(range(0, x.size - n + 1, h))
    if starts[-1] != x.size - n:
        starts.append(x.size - n)
    return [(s, s + n, x[s:s + n]) for s in starts]


def _consecutive(scores: list[float], ix: int, floor: float) -> int:
    count = 1
    for j in range(ix - 1, -1, -1):
        if scores[j] < floor: break
        count += 1
    for j in range(ix + 1, len(scores)):
        if scores[j] < floor: break
        count += 1
    return count


def augment(encoder: Any, enroll_emb: np.ndarray, streams: dict[str, np.ndarray], sr: int,
            cfg: ASEConfig) -> dict[str, Any]:
    """Return privacy-safe audit data plus private ``embedding`` for same-UID use only."""
    e0 = _norm(enroll_emb)
    candidates: list[tuple[str, int, int, np.ndarray]] = []
    for source, wav in streams.items():
        if source == "peak":
            continue
        candidates += [(source, a, b, w) for a, b, w in windows(wav, sr, cfg)]
    if not candidates:
        return {"applied": False, "reason": "no_candidate_windows", "embedding": e0}
    embs = encoder.embed_batch([x[3] for x in candidates], sr)
    scores = [_cos(e0, x) for x in embs]
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    best_i = order[0]
    second = scores[order[1]] if len(order) > 1 else -1.0
    source = candidates[best_i][0]
    source_ix = [i for i, c in enumerate(candidates) if c[0] == source]
    local_scores = [scores[i] for i in source_ix]
    local_pos = source_ix.index(best_i)
    support = _consecutive(local_scores, local_pos, scores[best_i] - cfg.support_margin)
    margin = scores[best_i] - second
    trusted = (scores[best_i] >= cfg.seed_abs and margin >= cfg.top2_margin and
               support >= cfg.min_consecutive)
    audit = {
        "applied": bool(trusted),
        "reason": "trusted_keyframe" if trusted else "untrusted_keyframe",
        "keyframe": {"source": source, "start_sec": round(candidates[best_i][1] / sr, 4),
                      "end_sec": round(candidates[best_i][2] / sr, 4),
                      "sim_e0": round(scores[best_i], 6), "top2_margin": round(margin, 6),
                      "consecutive_support": support},
        "config": asdict(cfg),
    }
    audit["embedding"] = _norm(cfg.lam * e0 + (1.0 - cfg.lam) * _norm(embs[best_i])) if trusted else e0
    return audit


def augmented_score(encoder: Any, embedding: np.ndarray, streams: dict[str, np.ndarray], sr: int) -> float:
    """Robust score: mean of top two stream similarities, never an isolated window maximum."""
    xs = [w for name, w in streams.items() if name != "peak"]
    if not xs: return -1.0
    sims = sorted((_cos(embedding, x) for x in encoder.embed_batch(xs, sr)), reverse=True)
    return float(sum(sims[:min(2, len(sims))]) / min(2, len(sims)))
