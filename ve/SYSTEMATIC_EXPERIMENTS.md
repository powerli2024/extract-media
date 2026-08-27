# Systematic KWS / gate / extraction experiments

目标口径只有一个：`score=(RR_neg + 1-CER_pos_micro)/2`。正样本被拒、提取失败或 ASR 最终失败时按该条 `CER=1`；宏平均 CER 只作诊断。

未知测试集上不存在由 DatasetA 离线实验“保证绝对最高分”的方法。本流程通过固定 ASR、重复分层留出、保守尾部分数、严格 coverage 和最终配对 bootstrap，降低多臂搜索造成的过拟合。

## 1. Gate screen: KWS × threshold policy

先固定下游为原始 CMD mix。所有 KWS 候选共享一份全正样本 ASR，分别重算 `mix/d1_spk1/d1_spk2` 声纹分数；比较 `max`、`strict_rescue`、`mix`。部署阈值取重复训练折最优阈值的中位数，不取全量同集 oracle。

```bash
cd /root/extract-main/ve
export KWS_CANDIDATES='mixed=/root/autodl-tmp/mixed_best/mixed_best;dae_se=/root/autodl-tmp/EXP02_DAE_THEN_MossFormer2_SE_48K;mmsfa=/root/autodl-tmp/mmsfa_dae_best_threshold/sep_3'
export EXP_ROOT=/root/autodl-tmp/ve_systematic_v1
export SEP_REUSE_ROOT=/root/autodl-tmp/ve_sep_cache/sep_streams
export STRICT_SEP_REUSE=1
SEEDS=200 HOLDOUT_FRAC=0.30 THRESHOLD_MODES=global,lang_split TOP_K=3 ./run_systematic_gate_screen.sh
```

### 0.1 补齐 `ve_sep_cache/reports/asr_cer`（标准全正样本 ASR）

`SEP_REUSE_ROOT` 只复用 `d1` 波形，不会产生正式 CER 所需的 ASR。缓存准备完成且已有对应 `results/all_results.jsonl` 与 `manifest/samples.jsonl` 后，运行：

```bash
cd /root/extract-main/ve
source .env_ve
conda activate ve
PYTHON_BIN="${PYTHON_BIN:-python}"

VE_OUT=/root/autodl-tmp/ve_sep_cache \
ASR_MODEL_DIR=/root/autodl-tmp/Qwen3-ASR-1.7B \
ASR_RESUME=0 \
PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/asr_cer.py \
  --ve-out /root/autodl-tmp/ve_sep_cache \
  --out-dir /root/autodl-tmp/ve_sep_cache/reports/asr_cer \
  --model-dir /root/autodl-tmp/Qwen3-ASR-1.7B \
  --device cuda:0 --no-resume --require-accepted-ok \
  2>&1 | tee /root/autodl-tmp/ve_sep_cache/reports/asr_cer/log.txt
```

结果保存为 `/root/autodl-tmp/ve_sep_cache/reports/asr_cer/asr_results.jsonl`，日志为同目录 `log.txt`。只有当该缓存目录的 `results/all_results.jsonl` 与 ASR 音频、解码配置一致时才可复用；正式候选更换提取音频或解码配置后必须全量重跑。

完成 `./run_next_lift.sh submit` 后必须对 overlay 显式严格计分：

```bash
mkdir -p /root/autodl-tmp/ve_sep_cache/reports/final_eval_submit
VE_OUT=/root/autodl-tmp/ve_sep_cache \
PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/final_evaluate.py \
  --ve-out /root/autodl-tmp/ve_sep_cache \
  --decisions /root/autodl-tmp/ve_sep_cache/reports/lift_overlay/submit_rows.jsonl \
  --asr /root/autodl-tmp/ve_sep_cache/reports/asr_cer/asr_results.jsonl \
  --out-dir /root/autodl-tmp/ve_sep_cache/reports/final_eval_submit \
  --strict 2>&1 | tee /root/autodl-tmp/ve_sep_cache/reports/final_eval_submit/log.txt
```

查看 `ranking/gate_ranking.md`。只有 coverage 完整且相对冻结基线的配对分差 `p05>0` 才入围；若没有入围项，保留仓库 `locked_thr.json`。

### 1.1 ASE-PVAD rescue 专用校准

ASE 影子全量与同 manifest 的全正样本 ASR 完成后，使用专用优化器；普通 `optimize_gate_for_score.py` 不支持 PVAD rescue 合同。

```bash
PVAD_AUDIT=/root/autodl-tmp/pvad_full_audit_d1_<tag>
ALL_POS=/root/autodl-tmp/pvad_all_pos_raw_v1
PVAD_OPT=/root/autodl-tmp/pvad_rescue_opt_v1

python scripts/optimize_pvad_rescue.py \
  --decisions "$PVAD_AUDIT/results/all_results.jsonl" \
  --asr-all-pos "$ALL_POS/reports/asr_cer/asr_results.jsonl" \
  --out-dir "$PVAD_OPT" \
  --threshold-modes global,lang_split \
  --holdout-frac 0.30 --seeds 500 \
  --group-field enroll_wav --strict
```

只有输出 `frozen_threshold.json` 且 `paired_delta.p05>0` 才能进入 `rescue_only`。运行时直接传阈值文件，不手抄数值：

```bash
PVAD_ASE=1 PVAD_MODE=rescue_only \
PVAD_CONFIG=./configs/pvad_ase_rescue_candidate.yaml \
PVAD_DECISION_THR_FILE="$PVAD_OPT/frozen_threshold.json" \
STRICT_ENROLL=1 STRICT_EVAL=1 ASR_RESUME=0 LIMIT=0 \
./run_all.sh
```

阈值文件中的配置哈希、聚合口径与当前运行不一致时启动即失败。全量同集最优值只用于诊断，部署值取重复训练折最优阈值的中位数。

## 2. Full finalists: KWS/gate × downstream arm

最多带 2--3 个第一阶段组合进入完整 ASR。每个条目格式为 `name|KWS目录|gate_score_opt目录|policy`。脚本会对每个下游臂强制处理全部正样本，再用该臂的真实 ASR 字符错误重新选择 `global/lang_split` 阈值；不会把 mix 最优阈值直接套给所有 TSE。

```bash
export FINALISTS='mixed_max|/root/autodl-tmp/mixed_best/mixed_best|/root/autodl-tmp/ve_systematic_v1/gate/mixed/reports/gate_score_opt|max;dae_rescue|/root/autodl-tmp/EXP02_DAE_THEN_MossFormer2_SE_48K|/root/autodl-tmp/ve_systematic_v1/gate/dae_se/reports/gate_score_opt|strict_rescue'
export PIPELINES='mix,sep_route,adaptive_route,wesep,ps4,cond_tasnet'
export EXP_ROOT=/root/autodl-tmp/ve_systematic_v1
RUN_CMD_SE=1 \
CMD_SE_ARMS=raw:raw:best,raw:se48k:best,se48k:raw:best,se48k:se48k:best,raw:se48k:better,se48k:se48k:better \
BASELINE_PIPELINE=sep_route \
BOOTSTRAP_REPLICATES=5000 \
./run_systematic_finalists.sh
```

第一个 finalist 的 `BASELINE_PIPELINE` 会额外用仓库冻结 `locked_thr.json` 生成真实基线，并作为 bootstrap 的第一项。正式候选必须满足：全量、all-positive ASR 全部 `ok`、`ASR_RESUME=0`、final coverage errors 为 0。最终先看实际官方分，再要求相对冻结基线的配对 bootstrap 分差 `p05>0`；同时检查中英文切片。若最高分臂没有稳定增益，选择基线或结构更简单、延迟更低的臂。确认赢家后，再用其 `gate_opt/recommended_thr.json` 做一次标准 `run_all.sh` 全链路复核与延迟测试。

## 3. Controlled ablations

- `RUN_CMD_SE=1` 会正式评测 `gate(raw/se48k) × ASR音频(raw/se48k) × best/better`。例如 `raw:se48k:best` 固定 raw 门控、只替换 SE 音频，可隔离识别收益；`se48k:raw:best` 只改变门控，可隔离 RR/FRR 变化；`se48k:se48k:best` 是完整 SE 方案。各臂独立重算 RR、FRR、CER。`run_all.sh` 中单独的 `CMD_SE=1` 仍只是生成候选，不应与正式矩阵混淆。
- `EXTRA_REJECT=0` 是主线。额外拒识只有在同一严格评测中提高官方分且 bootstrap 稳定时才开启。
- AS-Norm、窗口最大值、VAD 与新编码器会改变分数尺度，必须各自重新优化阈值，禁止复用其他臂的阈值。
- Wesep/PS4/Cond-TasNet 先用小规模冒烟验证依赖和输出，再跑全量；冒烟分数不进入排名。

## 4. Hidden-test safeguards

- 保留至少一个未参与日常调参的同分布批次作一次性验收；若已经反复查看 DatasetA 全量结果，DatasetA 不能再视为真正盲测集。
- 记录数据版本、KWS 根目录、模型权重哈希、阈值文件和代码提交；测试集上不再重新标定阈值。
- 优先选择跨 seed 阈值范围窄、中英文都不退化、依赖失败可回退的方案，而不是只领先万分位但方差很大的方案。
