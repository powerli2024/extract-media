from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import presence_encoder as pe  # noqa: E402
import run_extract  # noqa: E402


def test_campplus_does_not_fall_back_to_eres_dir(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake(*, model_dir, device):
        seen.update(model_dir=Path(model_dir), device=device)
        return "camp"

    monkeypatch.setattr(pe, "CAMPlusEncoder", fake)
    camp = tmp_path / "camp"
    assert pe.create_presence_encoder(
        "campplus", eres_dir=tmp_path / "eres", campplus_dir=camp, device="cpu"
    ) == "camp"
    assert seen["model_dir"] == camp


def test_ecapa_and_vblink100_aliases_use_separate_local_models(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []

    def fake(model_dir, *, name, device):
        calls.append((Path(model_dir), name, device))
        return name

    monkeypatch.setattr(pe, "WespeakerLocalEncoder", fake)
    ecapa = tmp_path / "ecapa"
    v100 = tmp_path / "v100"
    assert pe.create_presence_encoder(
        "ecapa_tdnn", ecapa_presence_dir=ecapa, device="cpu"
    ) == "ecapa1024_lm_voxceleb"
    assert pe.create_presence_encoder(
        "vblink2_samresnet100", vblink100_dir=v100, device="cuda:0"
    ) == "vblink2_samresnet100"
    assert calls == [
        (ecapa, "ecapa1024_lm_voxceleb", "cpu"),
        (v100, "vblink2_samresnet100", "cuda:0"),
    ]


def test_run_extract_passes_dedicated_campplus_dir_to_veto(
    monkeypatch, tmp_path: Path
) -> None:
    seen = {}

    def fake(backend, **kwargs):
        seen.update(backend=backend, **kwargs)
        return "veto"

    monkeypatch.setattr(run_extract, "create_presence_encoder", fake)
    args = Namespace(
        veto_backend="campplus",
        eres_dir=tmp_path / "eres",
        spk_chs_dir=tmp_path / "resnet",
        campplus_dir=tmp_path / "camp",
        device="cpu",
    )
    assert run_extract.create_veto_encoder(args) == "veto"
    assert seen["campplus_dir"] == tmp_path / "camp"
    assert seen["campplus_dir"] != seen["eres_dir"]
