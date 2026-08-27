from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pvad_self_augment import ASEConfig, augment, augmented_score

class FakeEncoder:
    def embed_batch(self, xs, sr):
        return [np.array([float(np.mean(x)), 1.0], dtype=np.float32) for x in xs]

def test_trusted_window_and_no_cross_call_state() -> None:
    e = np.array([1.0, 1.0], dtype=np.float32)
    streams = {"mix": np.array([0, 0, 1, 1], dtype=np.float32)}
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

def test_consecutive_support_allows_small_neighbor_score_change() -> None:
    e = np.array([1.0, 1.0], dtype=np.float32)
    # The best window and its overlapping neighbor are close but not identical.
    streams = {"mix": np.array([.7, .7, .8, .8, .8, .8], dtype=np.float32)}
    out = augment(FakeEncoder(), e, streams, 1,
                  ASEConfig(win_sec=2, hop_sec=1, top2_margin=0, min_consecutive=2,
                            support_margin=.05))
    assert out["applied"]

def test_top2_excludes_overlapping_temporal_support() -> None:
    e = np.array([1.0, 1.0], dtype=np.float32)
    # [0,1] overlaps the best [1,1] window, so it is support rather than top2.
    streams = {"mix": np.array([0, 0, 1, 1], dtype=np.float32)}
    out = augment(FakeEncoder(), e, streams, 1,
                  ASEConfig(win_sec=2, hop_sec=1, top2_margin=.1, min_consecutive=1))
    assert out["applied"]
    assert out["keyframe"]["top2_margin"] > .1

def test_allowed_sources_prevents_unapproved_stream_selection() -> None:
    e = np.array([1.0, 1.0], dtype=np.float32)
    streams = {"mix": np.ones(4, dtype=np.float32), "d1_spk1": np.zeros(4, dtype=np.float32)}
    out = augment(FakeEncoder(), e, streams, 1,
                  ASEConfig(win_sec=2, hop_sec=1, top2_margin=0, min_consecutive=1,
                            allowed_sources=("d1_spk1",)))
    assert out["keyframe"]["source"] == "d1_spk1"


def test_allowed_sources_also_limits_augmented_score() -> None:
    e = np.array([1.0, 1.0], dtype=np.float32)
    streams = {"mix": np.ones(4, dtype=np.float32), "d1_spk1": np.zeros(4, dtype=np.float32)}
    all_score = augmented_score(FakeEncoder(), e, streams, 1)
    approved_score = augmented_score(FakeEncoder(), e, streams, 1, ("d1_spk1",))
    assert approved_score != all_score


def test_invalid_ase_config_is_rejected() -> None:
    import pytest
    with pytest.raises(ValueError, match="lambda"):
        ASEConfig(lam=1.1)
