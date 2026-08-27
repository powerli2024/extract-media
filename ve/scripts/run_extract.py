#!/usr/bin/env python3
"""VE 端到端：Presence 拒识 → 提取方案之一。

PIPELINE / --tse-backend:
  ps4         — PS4 BSRNN（默认）
  wesep_bsrnn — WeSep 官方 bsrnn_ecapa_vox1
  sep_route   — MossFormer 分离 + enroll 声纹选路（强制 use_sep）
  adaptive_route — 分离流声纹显著优于 mix 时才选分离流（强制 use_sep）
  mix         — CMD mix 直通 ASR（不做 TSE）

产物:
  VE_OUT/extracted/{split}/{uid}__{routed_stream}.wav（sep_route/adaptive_route）
  VE_OUT/results/{split}_results.jsonl
  VE_OUT/reports/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

from audio_io import load_audio, save_audio
from calibrate_presence import stratified_limit
from paths import (
    default_campplus_dir,
    default_cohort_dir,
    default_eres2net_dir,
    default_ps4_weights,
    default_spk_chs_dir,
    default_test_cohort_dir,
    default_ve_out,
    default_wesep_dir,
    ensure_dir,
    setup_sys_path,
)
from presence_encoder import create_presence_encoder
from presence_gate import PresenceGate, try_create_onnx_separator
from presence_thr import load_thr_file, thr_for_sample
from report_ve import write_run_reports
from tse_factory import create_tse


def load_pvad_config(path: Path | None) -> dict[str, Any]:
    """Read flat P0 YAML without adding a runtime dependency."""
    out: dict[str, Any] = {"win_sec": 1.0, "hop_sec": 0.2, "lambda": .10, "seed_abs": 0.0,
                           "top2_margin": .04, "min_consecutive": 2, "support_margin": 0.0,
                           "gray_low": .05, "gray_high": .05, "allowed_sources": ""}
    if path is None:
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.split("#", 1)[0].strip()
        if ":" not in raw: continue
        key, value = (x.strip() for x in raw.split(":", 1))
        if key == "allowed_sources":
            out[key] = value
        elif key in out:
            out[key] = int(float(value)) if key == "min_consecutive" else float(value)
    return out


def load_pvad_decision_thresholds(
    path: Path | None, scalar: float | None
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    """Load a scalar or frozen language-aware PVAD threshold contract."""
    if path is not None and scalar is not None:
        raise ValueError("PVAD decision threshold scalar and file are mutually exclusive")
    if path is None:
        if scalar is None:
            return None, None
        value = float(scalar)
        if not -1.0 <= value <= 1.0:
            raise ValueError("PVAD decision threshold must be in [-1, 1]")
        return {"zh": value, "en": value, "default": value}, None
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("thr_by_lang") or data.get("thresholds")
    if not isinstance(raw, dict):
        value = data.get("pvad_decision_thr")
        if value is None:
            raise ValueError("PVAD threshold file has no thr_by_lang/thresholds/pvad_decision_thr")
        raw = {"zh": value, "en": value, "default": value}
    out = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    if "default" not in out and "zh" in out:
        out["default"] = out["zh"]
    if "zh" not in out and "default" in out:
        out["zh"] = out["default"]
    if "en" not in out and "default" in out:
        out["en"] = out["default"]
    if set(out) < {"zh", "en", "default"} or any(not -1.0 <= v <= 1.0 for v in out.values()):
        raise ValueError(f"invalid PVAD language thresholds: {out}")
    return out, data


def pvad_rescue_reason_allowed(reason: str) -> bool:
    """PVAD may rescue the frozen Presence reject, never an independent veto."""
    return str(reason or "") == "speaker_absent"

setup_sys_path()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_backend(name: str) -> str:
    b = (name or "ps4").lower().strip()
    aliases = {
        "ps4": "ps4",
        "ps4_bsrnn": "ps4",
        "bsrnn": "ps4",
        "wesep": "wesep_bsrnn",
        "wesep_bsrnn": "wesep_bsrnn",
        "wesep_bsrnn_ecapa": "wesep_bsrnn",
        "sep_route": "sep_route",
        "mossformer": "sep_route",
        "route": "sep_route",
        "sep": "sep_route",
        "adaptive_route": "adaptive_route",
        "adaptive": "adaptive_route",
        "mix_sep_route": "adaptive_route",
        "cond_tasnet": "cond_tasnet",
        "condtasnet": "cond_tasnet",
        "tasnet": "cond_tasnet",
        "cond-tasnet": "cond_tasnet",
        "mix": "mix",
        "passthrough": "mix",
        "mix_passthrough": "mix",
        "cmd": "mix",
        "none": "mix",
    }
    if b not in aliases:
        raise SystemExit(
            f"未知 --tse-backend={name!r}；可选: ps4 | wesep_bsrnn | sep_route | adaptive_route | mix | cond_tasnet"
        )
    return aliases[b]


def extracted_filename(uid: str, backend: str, meta: dict[str, Any] | None) -> str:
    """让分离选路产物携带实际选中的流名，避免 spk1/spk2 来源丢失。"""
    if backend in {"sep_route", "adaptive_route"}:
        stream = str((meta or {}).get("routed_stream") or "")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stream).strip("._")
        if safe:
            return f"{uid}__{safe}.wav"
    return f"{uid}.wav"


def waveform_fingerprint(wav: Any, sr: int) -> str:
    """格式无关的 CMD 波形指纹，避免误复用别的 CMD 三流。"""
    import numpy as np

    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    # sep_streams 写 WAV 时可能经历 16-bit 容器量化；14-bit 使同一 CMD 的
    # 源文件/缓存文件稳定一致，同时远高于“错误 UID 音频”会出现的差异尺度。
    q = np.rint(np.clip(x, -1.0, 1.0) * 8191.0).astype("<i2", copy=False)
    h = hashlib.sha256()
    h.update(f"sr={int(sr)};n={len(q)};".encode())
    h.update(q.tobytes())
    return h.hexdigest()


def load_reusable_d1_streams(
    root: Path, split: str, uid: str, cmd: Any, sr: int
) -> tuple[dict[str, Any] | None, str]:
    """仅 mix 指纹与本轮 CMD 一致时，安全返回 d1 的三个缓存流。"""
    base = Path(root) / "d1" / str(split) / str(uid)
    need = {name: base / f"{name}.wav" for name in ("mix", "d1_spk1", "d1_spk2")}
    missing = [name for name, p in need.items() if not p.is_file()]
    if missing:
        return None, f"missing:{','.join(missing)}"
    try:
        mix, mix_sr = load_audio(need["mix"])
        if waveform_fingerprint(mix, mix_sr) != waveform_fingerprint(cmd, sr):
            return None, "mix_fingerprint_mismatch"
        streams: dict[str, Any] = {"mix": mix}
        for name in ("d1_spk1", "d1_spk2"):
            streams[name], stream_sr = load_audio(need[name])
            if int(stream_sr) != int(sr):
                return None, f"sr_mismatch:{name}:{stream_sr}!={sr}"
        return streams, "hit"
    except Exception as e:  # noqa: BLE001
        return None, f"read_error:{type(e).__name__}:{e}"


def sep_cache_coverage(
    rows: list[dict[str, Any]], sep_root: Path, depth: int
) -> dict[str, Any]:
    """核验中间轨在 pos/neg × accept/reject 四类中的完整性。"""
    groups: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    for row in rows:
        split = str(row.get("split") or "?")
        decision = "accept" if str(row.get("decision")) == "accept" else "reject_or_error"
        key = f"{split}_{decision}"
        block = groups.setdefault(key, {"n": 0, "mix_saved": 0, "d1_pair_saved": 0})
        block["n"] += 1
        base = sep_root / f"d{depth}" / split / str(row.get("uid"))
        has_mix = (base / "mix.wav").is_file()
        has_pair = (base / "d1_spk1.wav").is_file() and (base / "d1_spk2.wav").is_file()
        block["mix_saved"] += int(has_mix)
        block["d1_pair_saved"] += int(has_pair)
        # 分离器报错时仍应保留 mix；否则要求完整 d1 双轨。
        valid = has_mix and (bool(row.get("sep_failed")) or has_pair)
        if not valid:
            missing.append({
                "uid": str(row.get("uid")), "split": split,
                "decision": str(row.get("decision")), "dir": str(base),
            })
    return {
        "sep_root": str(sep_root), "depth": depth,
        "groups": groups, "missing_count": len(missing), "missing": missing[:100],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VE Presence-gated 提取")
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--splits", default="pos,neg")
    p.add_argument("--presence-backend", default="eres2netv2")
    p.add_argument("--eres-dir", type=Path, default=None)
    p.add_argument("--spk-chs-dir", type=Path, default=None)
    p.add_argument("--presence-thr", type=float, default=-1.0, help="<0 则读校准文件")
    p.add_argument("--thr-file", type=Path, default=None)
    p.add_argument("--use-sep", action="store_true", help="Presence 用 MossFormer（默认 depth=1）")
    p.add_argument(
        "--sep-depth",
        type=int,
        default=-1,
        help="0=不分离 1=一次 2+=级联多次；-1 表示由 --use-sep / sep_route 决定",
    )
    p.add_argument(
        "--save-sep-wavs",
        action="store_true",
        help="保存分离中间轨到 VE_OUT/sep_streams/d{depth}/{split}/{uid}/",
    )
    p.add_argument(
        "--strict-sep-wavs",
        action="store_true",
        help="保存中间轨时要求所有样本均有 mix 与 d1 双轨（separator 失败例外），否则返回非零",
    )
    p.add_argument(
        "--reuse-sep-root", type=Path, default=None,
        help="复用已有 sep_streams 根；只有缓存 mix 与本轮 CMD 波形指纹一致时才使用 d1 三流",
    )
    p.add_argument(
        "--strict-reuse-sep", action="store_true",
        help="缓存未命中时不允许回退重新分离，直接失败",
    )
    p.add_argument(
        "--tse-backend",
        default="ps4",
        help="ps4 | wesep_bsrnn | sep_route | adaptive_route | mix | cond_tasnet",
    )
    p.add_argument("--cond-tasnet-ckpt", type=Path, default=None)
    p.add_argument("--ecapa-dir", type=Path, default=None)
    p.add_argument("--tasnet-chunk-sec", type=float, default=4.0)
    p.add_argument("--route-min-gain", type=float, default=0.03,
                   help="adaptive_route 选择分离流所需的最小声纹增益")
    p.add_argument("--ps4-weights", type=Path, default=None)
    p.add_argument("--wesep-dir", type=Path, default=None, help="兼容旧参数")
    p.add_argument("--wesep-model-dir", type=Path, default=None)
    p.add_argument("--wesep-language", default="english")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-tse", action="store_true", help="只跑 Presence（调试）")
    p.add_argument("--force-extract", action="store_true", help="忽略门控强制提取（对照）")
    p.add_argument("--write-reject-debug-wav", action="store_true")
    p.add_argument("--cohort-dir", type=Path, default=None)
    p.add_argument("--test-cohort-dir", type=Path, default=None)
    p.add_argument(
        "--enroll-znorm",
        action="store_true",
        help="enroll Z-Norm；thr 含 score_norm 时也会自动开",
    )
    p.add_argument("--test-znorm", action="store_true")
    p.add_argument("--asnorm", action="store_true")
    p.add_argument("--no-enroll-znorm", action="store_true", help="兼容旧开关，强制 raw")
    p.add_argument("--no-score-norm", action="store_true")
    p.add_argument("--cohort-per-spk", type=int, default=2)
    p.add_argument("--cohort-max-files", type=int, default=400)
    p.add_argument("--test-cohort-max-files", type=int, default=500)
    p.add_argument("--cohort-seed", type=int, default=0)
    p.add_argument("--znorm-eps", type=float, default=1e-3)
    p.add_argument(
        "--enroll-vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enroll 能量 VAD 裁剪静音后再 embed（默认开）",
    )
    p.add_argument("--enroll-vad-max-sec", type=float, default=4.0)
    p.add_argument(
        "--cmd-windows",
        default="off",
        help="off | slide | energy：CMD 滑窗/能量段打分；默认 ASR 用 argmax 窗",
    )
    p.add_argument("--win-sec", type=float, default=0.8)
    p.add_argument("--hop-sec", type=float, default=0.4)
    p.add_argument("--win-pad-ms", type=float, default=80.0)
    p.add_argument(
        "--stream-policy",
        default="max",
        choices=("max", "mix", "strict_rescue"),
        help="Presence 流融合：当前 max、仅 mix、或 mix 主判+严格分离救援",
    )
    p.add_argument("--rescue-high-margin", type=float, default=0.08)
    p.add_argument("--rescue-floor-margin", type=float, default=0.10)
    p.add_argument("--rescue-dominance", type=float, default=0.05)
    p.add_argument(
        "--asr-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="滑窗开启时是否把 mix ASR 裁到 argmax 窗（T2 默认开；T2b 用 --no-asr-crop）",
    )
    p.add_argument(
        "--veto-backend",
        default="",
        help="灰区第二路否决编码器（如 campplus）；空=关闭。只否决不救援",
    )
    p.add_argument("--campplus-dir", type=Path, default=None)
    p.add_argument("--veto-margin", type=float, default=0.12)
    p.add_argument("--veto-gray", type=float, default=0.10)
    p.add_argument(
        "--veto-windows",
        action="store_true",
        help="灰区且次优窗明显低于最优窗时否决",
    )
    p.add_argument("--pvad-ase", action="store_true", help="启用无训练 ASE-PVAD P0；默认关闭")
    p.add_argument("--pvad-mode", choices=("audit_only", "rescue_only", "bidirectional_gray"), default="audit_only")
    p.add_argument("--pvad-config", type=Path, default=None)
    p.add_argument(
        "--pvad-decision-thr", type=float, default=None,
        help="仅非 audit_only：经独立冻结折校准的 ASE 聚合分阈值；不可复用 Presence 阈值",
    )
    p.add_argument(
        "--pvad-decision-thr-file", type=Path, default=None,
        help="仅非 audit_only：optimize_pvad_rescue.py 生成的冻结阈值 JSON（支持 zh/en）",
    )
    return p.parse_args()


def resolve_sep_depth(args: argparse.Namespace, backend: str) -> int:
    if args.sep_depth >= 0:
        return int(args.sep_depth)
    if backend in ("sep_route", "adaptive_route") or args.use_sep:
        return 1
    return 0


def create_veto_encoder(args: argparse.Namespace) -> Any | None:
    veto_name = str(getattr(args, "veto_backend", "") or "").strip()
    if not veto_name:
        return None
    return create_presence_encoder(
        veto_name,
        eres_dir=args.eres_dir or default_eres2net_dir(),
        resnet_dir=args.spk_chs_dir or default_spk_chs_dir(),
        campplus_dir=args.campplus_dir or default_campplus_dir(),
        device=args.device,
    )


def main() -> int:
    args = parse_args()
    pvad_cfg = load_pvad_config(args.pvad_config) if args.pvad_ase else None
    pvad_config_sha = (hashlib.sha256(args.pvad_config.read_bytes()).hexdigest()
                       if args.pvad_ase and args.pvad_config is not None else None)
    try:
        pvad_decision_thresholds, pvad_threshold_contract = load_pvad_decision_thresholds(
            args.pvad_decision_thr_file, args.pvad_decision_thr
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"PVAD decision threshold invalid: {exc}") from exc
    pvad_impl_sha = None
    if args.pvad_ase:
        pvad_impl_sha = hashlib.sha256(
            Path(__file__).read_bytes()
            + (Path(__file__).resolve().parent / "pvad_self_augment.py").read_bytes()
        ).hexdigest()
    if args.pvad_ase and args.pvad_mode != "audit_only":
        if pvad_decision_thresholds is None:
            raise SystemExit("非 audit_only 的 PVAD 必须提供独立校准的 decision threshold")
        if pvad_threshold_contract is not None:
            expected_cfg = pvad_threshold_contract.get("config_sha256")
            if isinstance(expected_cfg, str) and expected_cfg != pvad_config_sha:
                raise SystemExit(
                    f"PVAD threshold config_sha256={expected_cfg} 与当前 config={pvad_config_sha} 不一致"
                )
            aggregation = pvad_threshold_contract.get("score_aggregation")
            if aggregation not in (None, "top2_mean_approved_streams"):
                raise SystemExit(f"PVAD threshold score_aggregation 不兼容: {aggregation}")
    backend = normalize_backend(args.tse_backend)
    sep_depth = resolve_sep_depth(args, backend)
    use_sep = sep_depth >= 1

    ve_out = (args.out_dir or default_ve_out()).resolve()
    ensure_dir(ve_out)
    samples_path = args.samples
    if samples_path is None:
        for cand in (
            ve_out / "manifest" / "samples.jsonl",
            default_ve_out() / "manifest" / "samples.jsonl",
            ve_out.parent / "manifest" / "samples.jsonl",
        ):
            if cand.is_file():
                samples_path = cand
                break
        if samples_path is None:
            samples_path = ve_out / "manifest" / "samples.jsonl"
    else:
        samples_path = samples_path.resolve()
    if not samples_path.is_file():
        raise SystemExit(f"找不到 samples.jsonl: {samples_path}（请先 build_manifest.py）")

    thr_file = args.thr_file or (
        ve_out / "reports" / "presence_calib" / "recommended_thr.json"
    )
    if not thr_file.is_file():
        alt = ve_out.parent / "reports" / "presence_calib" / "recommended_thr.json"
        if alt.is_file():
            thr_file = alt
        else:
            shared = Path(
                "/root/autodl-tmp/ve_presence_best/reports/presence_calib/recommended_thr.json"
            )
            if shared.is_file():
                thr_file = shared
            else:
                thr_file = (
                    default_ve_out() / "reports" / "presence_calib" / "recommended_thr.json"
                )

    thr_meta: dict[str, Any] = {}
    if args.presence_thr >= 0:
        thr_default = float(args.presence_thr)
    else:
        thr_default, thr_meta = load_thr_file(thr_file, 0.25)

    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    samples = [r for r in load_jsonl(samples_path) if r.get("split") in splits]
    if args.limit and args.limit > 0:
        samples = stratified_limit(samples, int(args.limit))

    want_mode = "raw"
    if not args.no_score_norm and not args.no_enroll_znorm:
        meta_mode = str(thr_meta.get("score_norm") or "raw")
        if args.asnorm or meta_mode == "asnorm":
            want_mode = "asnorm"
        elif args.test_znorm or meta_mode == "test_znorm":
            want_mode = "test_znorm"
        elif args.enroll_znorm or meta_mode == "enroll_znorm" or args.cohort_dir:
            want_mode = "enroll_znorm"
        if args.test_cohort_dir and want_mode == "enroll_znorm":
            want_mode = "asnorm"
        if args.test_cohort_dir and want_mode == "raw":
            want_mode = "test_znorm"
        if meta_mode == "raw" and not (
            args.asnorm or args.enroll_znorm or args.test_znorm or args.cohort_dir
        ):
            want_mode = "raw"

    thr_mode = thr_meta.get("thr_mode") or "global"
    from ve_tags import assert_thr_runtime_compatible

    for w in assert_thr_runtime_compatible(
        thr_meta, enroll_vad=bool(args.enroll_vad), strict=True
    ):
        print(f"[WARN] {w}", flush=True)
    if thr_meta:
        calibrated_policy = str(thr_meta.get("stream_policy") or "max")
        if calibrated_policy != str(args.stream_policy):
            raise SystemExit(
                f"[ERR] thr.stream_policy={calibrated_policy} 与当前 "
                f"stream_policy={args.stream_policy} 不一致；禁止串用阈值"
            )
        calibrated_sep = bool(thr_meta.get("use_sep", False))
        if calibrated_sep != bool(use_sep):
            raise SystemExit(
                f"[ERR] thr.use_sep={calibrated_sep} 与当前 use_sep={use_sep} 不一致；"
                "禁止串用阈值"
            )
        if calibrated_policy == "strict_rescue":
            for key, runtime in (
                ("rescue_high_margin", args.rescue_high_margin),
                ("rescue_floor_margin", args.rescue_floor_margin),
                ("rescue_dominance", args.rescue_dominance),
            ):
                calibrated = float(thr_meta.get(key, runtime))
                if abs(calibrated - float(runtime)) > 1e-9:
                    raise SystemExit(
                        f"[ERR] thr.{key}={calibrated} 与当前 {key}={runtime} 不一致；"
                        "禁止串用阈值"
                    )
    print(f"[INFO] VE_OUT={ve_out}")
    print(
        f"[INFO] samples={len(samples)} thr_default={thr_default} thr_mode={thr_mode} "
        f"backend={backend} sep_depth={sep_depth} save_sep={args.save_sep_wavs} "
        f"score_norm={want_mode} enroll_vad={bool(args.enroll_vad)}"
    )
    if thr_mode == "lang_split":
        print(f"[INFO] thr_by_lang={thr_meta.get('thr_by_lang')}", flush=True)
    if thr_meta.get("holdout"):
        print(f"[INFO] thr 来自 holdout 校准: {thr_meta.get('holdout')}", flush=True)
    print("[INFO] reject_policy=speaker_absent_only")

    enc = create_presence_encoder(
        args.presence_backend,
        eres_dir=args.eres_dir or default_eres2net_dir(),
        resnet_dir=args.spk_chs_dir or default_spk_chs_dir(),
        device=args.device,
    )
    sep = try_create_onnx_separator(peak=0.95, device=args.device) if use_sep else None
    if use_sep and sep is None:
        raise SystemExit(
            "sep_depth>=1 需要 MossFormer。请 ./download_moss_onnx.sh（extract/scripts/mossformer2_onnx.py）"
        )
    if backend in ("sep_route", "adaptive_route") and sep is None:
        raise SystemExit(f"PIPELINE={backend} 需要 MossFormer")

    score_norm = None
    if want_mode != "raw":
        from cohort_znorm import build_score_normalizer

        score_norm = build_score_normalizer(
            enc,
            mode=want_mode,  # type: ignore[arg-type]
            enroll_dir=(
                Path(args.cohort_dir) if args.cohort_dir else default_cohort_dir()
            )
            if want_mode in ("enroll_znorm", "asnorm")
            else None,
            test_dir=(
                Path(args.test_cohort_dir)
                if args.test_cohort_dir
                else default_test_cohort_dir()
            )
            if want_mode in ("test_znorm", "asnorm")
            else None,
            enroll_per_spk=int(args.cohort_per_spk),
            enroll_max_files=int(args.cohort_max_files),
            test_max_files=int(args.test_cohort_max_files),
            seed=int(args.cohort_seed),
            eps=float(args.znorm_eps),
        )

    veto_name = str(getattr(args, "veto_backend", "") or "").strip()
    veto_enc = create_veto_encoder(args)
    if veto_enc is not None:
        print(f"[INFO] veto encoder={veto_enc.name} margin={args.veto_margin}", flush=True)

    gate = PresenceGate(
        enc,
        thr=thr_default,
        use_sep=bool(sep),
        separator=sep,
        sep_depth=sep_depth if sep is not None else 0,
        score_normalizer=score_norm,
        enroll_vad=bool(args.enroll_vad),
        enroll_vad_max_sec=float(args.enroll_vad_max_sec),
        cmd_window_mode=str(args.cmd_windows or "off"),
        win_sec=float(args.win_sec),
        hop_sec=float(args.hop_sec),
        win_pad_ms=float(args.win_pad_ms),
        veto_encoder=veto_enc,
        veto_margin=float(args.veto_margin),
        veto_gray=float(args.veto_gray),
        veto_windows=bool(args.veto_windows),
        stream_policy=str(args.stream_policy),
        rescue_high_margin=float(args.rescue_high_margin),
        rescue_floor_margin=float(args.rescue_floor_margin),
        rescue_dominance=float(args.rescue_dominance),
    )
    actual_depth = gate.sep_depth
    print(
        f"[INFO] enroll_vad={gate.enroll_vad} max_sec={gate.enroll_vad_max_sec} "
        f"cmd_windows={gate.cmd_window_mode} veto_windows={gate.veto_windows} "
        f"stream_policy={gate.stream_policy}",
        flush=True,
    )

    extractor = None
    if not args.skip_tse:
        extractor = create_tse(
            backend,
            weights=args.ps4_weights or default_ps4_weights(),
            device=args.device,
            wesep_dir=args.wesep_dir or default_wesep_dir(),
            wesep_model_dir=args.wesep_model_dir,
            wesep_language=args.wesep_language,
            separator=sep,
            encoder=enc,
            cond_tasnet_ckpt=args.cond_tasnet_ckpt,
            ecapa_dir=args.ecapa_dir,
            tasnet_chunk_sec=float(args.tasnet_chunk_sec),
            route_min_gain=float(args.route_min_gain),
        )

    results_dir = ensure_dir(ve_out / "results")
    extracted_dir = ensure_dir(ve_out / "extracted")
    debug_dir = ensure_dir(ve_out / "debug_reject") if args.write_reject_debug_wav else None
    sep_root = (
        ensure_dir(ve_out / "sep_streams" / f"d{actual_depth}")
        if args.save_sep_wavs and actual_depth >= 1
        else None
    )
    reuse_sep_root = args.reuse_sep_root.resolve() if args.reuse_sep_root else None
    if reuse_sep_root is not None and not (reuse_sep_root / "d1").is_dir():
        raise SystemExit(f"--reuse-sep-root 缺少 d1/: {reuse_sep_root}")
    reuse_stats: dict[str, int] = {"hit": 0, "miss": 0, "fresh": 0}
    if reuse_sep_root is None:
        print("[SEP_REUSE] enabled=0 (no --reuse-sep-root; will run separator)", flush=True)
    else:
        print(
            f"[SEP_REUSE] enabled=1 root={reuse_sep_root} strict={bool(args.strict_reuse_sep)} "
            "validation=mix_waveform_fingerprint",
            flush=True,
        )

    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in splits}
    t_run0 = time.time()

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    iterator = (
        tqdm(samples, desc=f"extract:{backend}", unit="utt", mininterval=0.5)
        if tqdm is not None
        else samples
    )
    for i, it in enumerate(iterator):
        uid = it["uid"]
        split = it["split"]
        thr = thr_for_sample(it, default_thr=thr_default, thr_meta=thr_meta)
        t0 = time.time()
        rec: dict[str, Any] = {
            "uid": uid,
            "split": split,
            "id": it.get("id"),
            "label": it.get("label"),
            "lang": it.get("lang"),
            "wake_text": it.get("wake_text"),
            "cmd_text": it.get("cmd_text"),
            "enroll_wav": it.get("enroll_wav"),
            "cmd_wav": it.get("cmd_wav"),
            "presence_backend": enc.name,
            "tse_backend": None if extractor is None else extractor.name,
            "reject_policy": "speaker_absent_only",
            "pipeline": backend,
            "thr_mode": thr_mode,
        }
        try:
            enroll, sr = load_audio(it["enroll_wav"])
            cmd, _ = load_audio(it["cmd_wav"])
            save_dir = None
            if sep_root is not None:
                save_dir = sep_root / split / uid
            cached_streams = None
            cache_state = "fresh"
            cache_dir = None
            if reuse_sep_root is not None and actual_depth == 1:
                cached_streams, cache_state = load_reusable_d1_streams(
                    reuse_sep_root, split, uid, cmd, sr
                )
                if cached_streams is not None:
                    reuse_stats["hit"] += 1
                    cache_dir = reuse_sep_root / "d1" / split / uid
                    # 外部缓存不复制，既避免无意义 I/O，也保留唯一可追溯来源。
                    save_dir = None
                else:
                    reuse_stats["miss"] += 1
                    if args.strict_reuse_sep:
                        raise RuntimeError(f"strict reusable sep cache miss: {cache_state}")
            pr, streams, enroll_emb = gate.score_with_streams(
                enroll, cmd, enroll_key=uid, sr=sr, thr=thr, save_dir=save_dir,
                precomputed_streams=cached_streams, precomputed_sep_dir=cache_dir,
            )
            pvad_audit: dict[str, Any] | None = None
            if pvad_cfg is not None:
                # P0 is stateless: no embedding is persisted or reused across UIDs.
                # Any ASE fault is fail-open: it must never turn a valid frozen-gate
                # result into a pipeline error, including in audit-only mode.
                try:
                    from pvad_self_augment import ASEConfig, augment, augmented_score
                    cfg = ASEConfig(win_sec=float(pvad_cfg["win_sec"]), hop_sec=float(pvad_cfg["hop_sec"]),
                        lam=float(pvad_cfg["lambda"]), seed_abs=float(pvad_cfg["seed_abs"]),
                        top2_margin=float(pvad_cfg["top2_margin"]), min_consecutive=int(pvad_cfg["min_consecutive"]),
                        support_margin=float(pvad_cfg["support_margin"]),
                        allowed_sources=tuple(x.strip() for x in str(pvad_cfg["allowed_sources"]).split(",") if x.strip()))
                    pvad_audit = augment(enc, enroll_emb, streams, sr, cfg)
                    score_aug = augmented_score(enc, pvad_audit.pop("embedding"), streams, sr, cfg.allowed_sources)
                    pvad_decision_thr = None
                    if pvad_decision_thresholds is not None:
                        lang = str(it.get("lang") or "zh")
                        pvad_decision_thr = float(
                            pvad_decision_thresholds.get(lang, pvad_decision_thresholds["default"])
                        )
                    pvad_audit.update({"mode": args.pvad_mode, "score_static": round(float(pr.score), 6),
                        "score_aug": round(score_aug, 6), "config_sha256": pvad_config_sha,
                        "decision_thr": pvad_decision_thr,
                        "score_aggregation": "top2_mean_approved_streams",
                        "fallback_used": not bool(pvad_audit["applied"])})
                    gray = float(pvad_cfg["gray_low"] if pr.score < pr.thr else pvad_cfg["gray_high"])
                    pvad_audit["decision_eligible"] = bool(pr.score_norm == "raw" and abs(pr.score - pr.thr) <= gray)
                    pvad_audit["rescue_blocked_by_veto"] = bool(
                        args.pvad_mode == "rescue_only" and pr.reject
                        and not pvad_rescue_reason_allowed(pr.reason)
                    )
                    if (pvad_audit["applied"] and args.pvad_mode == "rescue_only" and pr.reject
                            and pvad_rescue_reason_allowed(pr.reason)
                            and pvad_audit["decision_eligible"] and score_aug >= pvad_decision_thr):
                        pr.score, pr.reject, pr.reason = score_aug, False, "pvad_ase_rescue"
                    elif (pvad_audit["applied"] and args.pvad_mode == "bidirectional_gray"
                            and pvad_audit["decision_eligible"]):
                        pr.score, pr.reject = score_aug, score_aug < pvad_decision_thr
                        pr.reason = "pvad_ase_gray_reject" if pr.reject else ""
                except Exception as e:
                    pvad_audit = {
                        "applied": False, "reason": "pvad_ase_exception",
                        "error_type": type(e).__name__, "error": str(e)[:500],
                        "mode": args.pvad_mode, "score_static": round(float(pr.score), 6),
                        "score_aug": round(float(pr.score), 6), "config_sha256": pvad_config_sha,
                        "decision_thr": (
                            pvad_decision_thresholds.get(str(it.get("lang") or "zh"), pvad_decision_thresholds["default"])
                            if pvad_decision_thresholds is not None else None
                        ),
                        "score_aggregation": "top2_mean_approved_streams",
                        "fallback_used": True, "decision_eligible": False,
                    }
            if cached_streams is None:
                reuse_stats["fresh"] += 1
            rec.update(pr.to_dict())
            if pvad_audit is not None:
                rec["pvad"] = pvad_audit
            rec["sep_source"] = "cache" if cached_streams is not None else "fresh"
            rec["sep_cache_validation"] = cache_state
            rec["presence_thr"] = thr
            rec["presence_ms"] = round((time.time() - t0) * 1000, 1)

            do_extract = (not pr.reject) or args.force_extract
            if args.skip_tse:
                do_extract = False

            if pr.reject and not args.force_extract:
                rec["decision"] = "reject"
                rec["reject_reason"] = pr.reason or "speaker_absent"
                rec["extracted_wav"] = None
                if debug_dir and extractor is not None:
                    try:
                        if backend in ("sep_route", "adaptive_route"):
                            dbg, meta = extractor.extract(
                                cmd,
                                enroll,
                                sr=sr,
                                streams=streams,
                                enroll_emb=enroll_emb,
                                preferred_stream=pr.best_stream,
                            )
                        else:
                            dbg, meta = extractor.extract(cmd, enroll, sr=sr)
                        dp = debug_dir / split / f"{uid}.wav"
                        save_audio(dp, dbg, sr)
                        rec["debug_tse_wav"] = str(dp)
                        rec["debug_tse_meta"] = meta
                    except Exception as e:
                        rec["debug_tse_error"] = str(e)
            elif do_extract and extractor is not None:
                t1 = time.time()
                try:
                    if backend in ("sep_route", "adaptive_route"):
                        out, meta = extractor.extract(
                            cmd,
                            enroll,
                            sr=sr,
                            streams=streams,
                            enroll_emb=enroll_emb,
                            preferred_stream=pr.best_stream,
                        )
                    else:
                        out, meta = extractor.extract(cmd, enroll, sr=sr)
                    if pr.best_window and backend == "mix" and getattr(args, "asr_crop", True):
                        from window_geom import crop_with_pad

                        out, wmeta = crop_with_pad(
                            out,
                            int(pr.best_window["start"]),
                            int(pr.best_window["end"]),
                            sr,
                            pad_ms=float(args.win_pad_ms),
                        )
                        meta = dict(meta or {})
                        meta["asr_crop"] = wmeta
                    elif pr.best_window and backend == "mix":
                        meta = dict(meta or {})
                        meta["asr_crop"] = {
                            "skipped": True,
                            "reason": "full_mix",
                            "best_start": int(pr.best_window["start"]),
                            "best_end": int(pr.best_window["end"]),
                        }
                    out_path = extracted_dir / split / extracted_filename(uid, backend, meta)
                    save_audio(out_path, out, sr)
                    rec["decision"] = "accept"
                    rec["reject_decision"] = False
                    rec["reject_reason"] = ""
                    rec["extracted_wav"] = str(out_path.resolve())
                    rec["extracted_stream"] = (meta or {}).get("routed_stream")
                    rec["tse_meta"] = meta
                    rec["tse_ms"] = round((time.time() - t1) * 1000, 1)
                except Exception as e:
                    rec["decision"] = "extract_error"
                    rec["extract_error"] = str(e)
                    rec["extract_traceback"] = traceback.format_exc(limit=5)
            else:
                rec["decision"] = "accept_no_tse" if not pr.reject else "reject"
            rec["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
        except Exception as e:
            rec["decision"] = "pipeline_error"
            rec["error"] = str(e)
            rec["traceback"] = traceback.format_exc(limit=8)
            rec["elapsed_ms"] = round((time.time() - t0) * 1000, 1)

        by_split.setdefault(split, []).append(rec)
        n_done = i + 1
        if n_done % 500 == 0 or n_done == len(samples):
            msg = (
                f"last={uid} "
                f"decision={rec.get('decision')} score={rec.get('presence_score')} "
                f"thr={rec.get('presence_thr')}"
            )
            if tqdm is not None:
                # 保留 tqdm 的单行刷新，不再用 tqdm.write 把进度条打散为多行。
                iterator.set_postfix_str(msg, refresh=False)
            else:
                print(f"[INFO] {n_done}/{len(samples)} {msg}", flush=True)

    all_rows: list[dict[str, Any]] = []
    for split, rows in by_split.items():
        path = results_dir / f"{split}_results.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                all_rows.append(r)
        print(f"[OK] {path} n={len(rows)}")

    pvad_rows = [r.get("pvad") for r in all_rows if isinstance(r.get("pvad"), dict)]
    pvad_summary = None
    if pvad_cfg is not None:
        pvad_summary = {
            "mode": args.pvad_mode,
            "config_sha256": pvad_config_sha,
            "implementation_sha256": pvad_impl_sha,
            "decision_thresholds": pvad_decision_thresholds,
            "decision_threshold_file": str(args.pvad_decision_thr_file) if args.pvad_decision_thr_file else None,
            "score_aggregation": "top2_mean_approved_streams",
            "n_rows": len(pvad_rows),
            "n_applied": sum(bool(r.get("applied")) for r in pvad_rows),
            "n_fallback": sum(bool(r.get("fallback_used")) for r in pvad_rows),
            "n_exceptions": sum(r.get("reason") == "pvad_ase_exception" for r in pvad_rows),
        }

    with (results_dir / "all_results.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = write_run_reports(
        ve_out / "reports",
        all_rows,
        meta={
            "presence_thr": thr_default,
            "thr_mode": thr_mode,
            "thr_by_lang": thr_meta.get("thr_by_lang"),
            "presence_backend": enc.name,
            "use_sep": actual_depth >= 1,
            "sep_depth": actual_depth,
            "save_sep_wavs": bool(sep_root),
            "score_norm": want_mode,
            "stream_policy": gate.stream_policy,
            "rescue_high_margin": gate.rescue_high_margin,
            "rescue_floor_margin": gate.rescue_floor_margin,
            "rescue_dominance": gate.rescue_dominance,
            "tse_backend": None if extractor is None else extractor.name,
            "pipeline": backend,
            "elapsed_sec": round(time.time() - t_run0, 2),
            "n_samples": len(all_rows),
            "ve_out": str(ve_out),
            "force_extract": args.force_extract,
            "skip_tse": args.skip_tse,
            "thr_file": str(thr_file) if thr_file else None,
            "pvad": pvad_summary,
            "reuse_sep_root": str(reuse_sep_root) if reuse_sep_root else None,
            "reuse_sep_stats": reuse_stats,
        },
    )
    coverage_root = reuse_sep_root if reuse_sep_root is not None else (ve_out / "sep_streams")
    reuse_display: Any = reuse_stats if reuse_sep_root is not None else "disabled"
    if sep_root is not None or reuse_sep_root is not None:
        coverage = sep_cache_coverage(all_rows, coverage_root, actual_depth)
        coverage_path = ve_out / "reports" / "sep_streams_coverage.json"
        coverage_path.write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[SEP_CACHE] root={coverage_root} groups={coverage['groups']} "
            f"missing={coverage['missing_count']} reuse={reuse_display} → {coverage_path}",
            flush=True,
        )
        if args.strict_sep_wavs and coverage["missing_count"]:
            print("[ERR] strict sep cache coverage failed", flush=True)
            return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] reports → {ve_out / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
