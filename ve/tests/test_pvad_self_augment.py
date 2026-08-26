from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pvad_self_augment import ASEConfig, augment

class FakeEncoder:
    def embed_batch(self, xs, sr):
        return [np.array([float(np.mean(x)), 1.0], dtype=np.float32) for x in xs]

def test_trusted_window_and_no_cross_call_state() -> None:
    e = np.array([1.0, 1.0], dtype=np.float32)
    streams = {"mix": np.array([0, 0, 1, 1, 1, 1], dtype=np.float32)}
    cfg = ASEConfig(win_sec=2, hop_sec=1, seed_abs=0, top2_margin=0, min_consecutive=1)
    a = augment(FakeEncoder(), e, streams, 1, cfg)
    b = augment(FakeEncoder(), e, streams, 1, cfg)
    assert a["applied"] and b["applied"]
    assert np.allclose(a["embedding"], b["embedding"])

def test_margin_rejects_ambiguous_windows() -> None:
    e = np.array([1.0, 1.0], dtype=np.float32)
    streams = {"mix": np.ones(6, dtype=np.float32)}
    out = augment(FakeEncoder(), e, streams, 1, ASEConfig(win_sec=2, hop_sec=1, top2_margin=.1))
    assert not out["applied"]
