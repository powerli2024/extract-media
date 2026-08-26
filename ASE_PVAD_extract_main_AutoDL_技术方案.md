# ASE-PVAD 与 extract-main 融合技术方案

> 文档状态：设计基线 v1.0  
> 编写日期：2026-08-26  
> 对应代码快照：`extract-main@0712a7c`  
> 目标环境：AutoDL，单卡 NVIDIA RTX 4090 24 GB（同时给出降级路径）  
> 论文：[Adaptive Speaker Embedding Self-Augmentation for Personal Voice Activity Detection with Short Enrollment Speech](https://arxiv.org/abs/2601.12769)，ICASSP 2026

## 1. 结论与实施建议

这篇论文适合用于改进 `extract-main/ve` 的 **Presence 门控**，但不应直接替换现有整条 KWS/TSE/ASR 流程。推荐分两期实施：

1. **P0：无需训练的 ASE 门控（优先）**。复用现有 CAM++/ERes2NetV2 编码器和 `sep_streams` 缓存，在 CMD 内选取可信目标说话人关键窗，形成一次性的增强声纹，再重新计算 Presence 证据。该阶段可以快速验证论文最核心的“混合语音反哺短注册声纹”思想。
2. **P1：完整三分类 PVAD（后续）**。实现论文的 Conformer + FiLM 帧级模型，输出非语音、非目标人声、目标人声三类后验；只有 P0 已显示稳定收益且训练数据合同满足时再投入训练。

默认上线策略必须保持保守：

- `PVAD_ASE=0` 时行为与当前版本完全一致；
- ASE/PVAD 仅产生门控分数和审计信息，**ASR 仍读取 raw best 音频**；
- 单条 CMD 失败、显存不足、没有可信关键窗或模型校验失败时，自动回退当前冻结门控；
- 不在彼此独立的评测样本间累积声纹状态，避免测试集信息泄漏和错误漂移；
- 只有严格覆盖、竞赛分和独立验证全部达到第 10 节标准后，才允许改变生产默认值。

## 2. 当前问题与计分目标

任务输入为一段较干净的注册人 KWS 音频和一段可能含其他人声、重叠人声及噪声的 CMD 音频。系统需要先判断 CMD 是否包含注册人：

- 负样本应拒识，指标为 `RR_neg`；
- 正样本应接收并识别注册人的命令；
- 正样本若被误拒、ASR 最终失败或缺结果，该样本按 `CER=1`；
- 正样本 CER 使用字符数加权的 micro CER，而不是逐句宏平均。

唯一正式目标为：

```text
score = 0.5 * RR_neg + 0.5 * (1 - CER_pos_micro)
```

因此，单纯提高 PVAD F1、声纹相似度或负样本 RR 都不能证明方案有效。任何更严格的拒识都会提高 RR，但也可能通过正样本误拒把 CER 推高；最终必须由 `scripts/final_evaluate.py --strict` 判定。

当前代码已经具备以下可复用能力：

- Presence 编码器统一接口：ERes2NetV2、CAM++、ECAPA-TDNN、VoxBlink2 SimAM-ResNet；
- MossFormer2 分离出的 `mix/d1_spk1/d1_spk2` 及跨实验缓存；
- `max`、`mix`、`strict_rescue` 三类选流策略；
- CMD-SE 对照与 `mix_top_se48k` 策略；
- 全正样本固定 ASR、阈值重复留出优化、严格覆盖检查和配对 bootstrap 排名；
- ASR 空结果重试与最终 coverage 审计。

`log9.txt` 中当前 DatasetA 候选结果如下。它们只用于确定本项目验收基线，不能视为未知测试集的保证值：

| 臂 | RR | CER micro | 竞赛分 | 相对冻结基线 paired delta p05 |
|---|---:|---:|---:|---:|
| `locked.sep_route` | 0.886076 | 0.374159 | 0.755958 | 0 |
| `strict_v2.cmd_se.se48kgate_raw_best` | 0.911392 | 0.371215 | **0.770089** | +0.001301 |

后者的优势下界较薄，必须通过独立 KWS 注册集、不同噪声/叠话条件和重复实验复核。

## 3. 论文方法摘要及适用边界

### 3.1 论文核心方法

论文面向短注册语音下的 Personal Voice Activity Detection（PVAD）：

- 使用预训练 CAM++ 从注册语音得到 192 维声纹；
- 混合语音使用 40 维 log-Mel 特征，25 ms 帧长、10 ms 帧移；
- 主干为 4 层 Conformer，通过 FiLM 注入目标声纹；
- 帧级标签为 `NS`（非语音）、`NTSS`（非目标人声）、`TSS`（目标人声）；
- Weighted Pairwise Loss 中 `TSS-NS`、`TSS-NTSS` 权重为 1，`NS-NTSS` 权重为 0.5；
- 自增强阶段以 1.0 s 窗、0.2 s 步长扫描混合语音，用 CAM++ 得到窗级声纹，选择与注册声纹余弦相似度最高且超过阈值的关键窗；
- 论文比较拼接与相加融合，整体上加法融合更好；
- 长期更新把当前关键窗、历史增强声纹和原始注册声纹进行残差式融合，论文使用 `lambda=0.1`，消融中 `0.05~0.1` 较优；
- 论文报告在其英文合成评测中，0.5 s 注册音频经过 5 次迭代可接近平均 7.4 s 完整注册音频的表现。

部署时统一采用 L2 归一化后的实现：

```text
e0        = normalize(encoder(enroll))
ek        = normalize(encoder(selected_key_window))
eavg(n)   = normalize(0.5 * (eaug(n-1) + ek))
eaug(n)   = normalize(lambda * e0 + (1-lambda) * eavg(n))
```

第一轮令 `eaug(0)=e0`。所有中间向量、余弦值、阈值和选择原因均写入 JSONL。

### 3.2 不能直接照搬的部分

论文实验与本项目存在显著域差异：

- 论文采用 LibriSpeech 英文语音合成三说话人混合，加入 MUSAN 0–20 dB 噪声；本项目包含中英文短命令、真实噪声和未知叠话结构；
- 论文优化 ACC、Recall、Precision、F1、AP，本项目优化 RR 与字符加权 CER 的组合分；
- 论文的长期更新假定同一目标用户有连续片段，本项目的离线样本通常彼此独立；
- 最大相似度关键窗可能来自非目标说话人，存在确认偏差。论文自身也显示，不保留原始注册声纹锚点的朴素迭代会随轮次退化；
- 论文原匿名链接在 2026-08-26 检查时返回 HTTP 410，但用户补充了可访问的 [ASE-PVAD GitHub 仓库](https://github.com/asdhajksdh/ASE-PVAD)。该仓库可用于核对网络和实验逻辑，但仍不是可直接上线的推理包；许可证、权重、路径和环境兼容性必须先完成审计。

因此，P0 首先验证自增强声纹对本项目门控是否有效；P1 的论文完整网络不能因为论文 F1 提升就直接上线。

### 3.3 上游 GitHub 代码审计结论

2026-08-26 对用户提供的 [README](https://github.com/asdhajksdh/ASE-PVAD/blob/main/readme.md)、环境和关键训练/模型文件进行静态核对，结论如下：

| 项目 | 上游现状 | 对本项目的处理 |
|---|---|---|
| 数据 | README 指向 LibriSpeech 与 MUSAN，并提供 train/test 数据准备 shell | 数据生成思路可参考；不能把其路径或英文域假定带入 DatasetA |
| 声纹 | README 同时提到 Resemblyzer 和 CAM++，CAM++ 部分基于 3D-Speaker | P0/P1 首选复用本仓已验证的 `CAMPlusEncoder`，不引入第二套 CAM++ 运行时 |
| 训练入口 | 提供 Eq.4 与 Eq.5–6 对应训练脚本 | 作为算法对照；重新封装 argparse/config、数据合同和 checkpoint manifest |
| 推理入口 | 仅提供 segment1、segment1–5 的研究测试脚本 | 不能直接接 `samples.jsonl`；需实现本方案的批量推理适配器 |
| 路径 | 关键训练脚本含大量 `/data/private/PVAD/...` 绝对路径 | 全部参数化，AutoDL 数据只落 `/root/autodl-tmp` |
| 关键窗 | 已核对训练数据读取代码采用余弦 `argmax` 选择混合窗 | 加入绝对阈值、top2 margin、连续性和双证据；避免无条件最大值造成负样本误接收 |
| 模型 | 模型文件包含 40→64 输入、192→64 FiLM、3 类输出和 4 次 Conformer 调用 | 先写结构等价单测；检查 4 层是否意外共享同一个 block 实例，再决定兼容旧权重或修正为独立层 |
| 长期融合 | 上游模型代码可见 `0.5*(history+selected)` 与 `0.1*enroll+0.9*average` | 与论文公式一致，但部署仍增加 L2 归一化、更新上限、状态隔离和回滚 |
| 训练轮数 | 论文写 100 epoch；已核对的一份上游训练脚本设置 200 epoch | 以验证集早停为准，同时记录差异；不能宣称脚本未经修改即可复现论文 |
| 环境 | `environment.yml` 为 Python 3.8、PyTorch 2.2.0/cu121，且同时列出 CUDA 12.1 组件和 `cudatoolkit=11.1.1` | 不导入当前 `ve` 环境；参考复现用独立环境，生产移植只添加最小依赖 |
| 权重 | 仓库根目录和 README 未提供明确的预训练 PVAD checkpoint/下载步骤 | P1 必须自行训练或取得作者权重并校验；没有权重时不得把 P1 写成可部署完成态 |
| 许可证 | 仓库根目录列表未见 LICENSE，README 也未声明许可证 | 在作者明确授权前，仅用于研究验证和结构参考；生产不能直接复制未授权代码 |

上游代码采用如下策略进入本项目：

1. 记录具体 commit SHA，不跟随浮动 `main`；
2. 先生成第三方代码清单和许可证结论；
3. 优先按论文公式在 `extract-main` 独立实现，测试输出与上游小样例对齐；
4. 若确需复制上游实现，仅 vendor 经过审计的最小模型文件，保留来源和修改记录；
5. 不允许运行上游脚本写死的训练/结果目录，也不允许其环境覆盖 `/root/miniconda3/envs/ve`。

## 4. 总体架构

```text
KWS enroll ──> enroll预处理 ──> speaker encoder ──> e0 ───────────────┐
                                                                        │
CMD raw ──> 分离缓存(mix/spk1/spk2) ──> 候选窗编码 ──> 安全选窗 ──> ek │
                                                                        v
                                                           自增强 eaug / PVAD
                                                                        │
                                           ┌────────────────────────────┴─────┐
                                           v                                  v
                                    Presence决策/拒识                    审计JSONL
                                           │
                            accept ─────────┴───────── reject
                              │                           │
                       raw best 音频 ASR              正样本 CER=1
                              │
                       字符加权 CER
```

关键原则是 **门控证据与 ASR 音频解耦**。已有实验表明 SE 可能改善 RR/FRR，却可能损伤 ASR，因此 ASE/PVAD 可以使用 raw、SE 或分离流形成门控证据，但接收后的 ASR 默认仍使用未经 SE 的 raw best。任何改变 ASR 输入的臂必须单独全量重跑，不可复用旧 ASR。

## 5. P0：无需训练的安全自增强门控

### 5.1 候选音频和窗

候选来源按以下集合生成：

```text
raw_mix, raw_d1_spk1, raw_d1_spk2
可选：se48k_mix, se48k_d1_spk1, se48k_d1_spk2
```

- 优先复用 `SEP_REUSE_ROOT/d1/<split>/<uid>/`，并校验原 CMD 波形指纹、采样率、长度和分离模型版本；
- 默认 1.0 s 窗、0.2 s hop，与论文一致；不足 1.0 s 时只做一次 padding 后编码；
- 对静音占比、削波率、RMS、有效语音时长做基础过滤；
- 每个候选保存 `source/start_sec/end_sec/sim_e0/rms/vad_ratio`，但不必保存全部窗音频；调试模式才写 wav。

### 5.2 安全选窗，避免“最大相似度即接收”

不能沿用无约束最大值。一个关键窗必须同时满足：

1. `sim(e0, ek) >= tau_seed_abs`；
2. top1 与 top2 的差值 `>= margin_top2`；
3. 至少连续 `min_consecutive` 个重叠窗达到支持阈值，或相邻窗后验具有一致性；
4. 若使用分离流，目标流相对另一路的优势 `>= margin_stream`；
5. 若启用双编码器确认，主编码器和 veto 编码器对目标存在性的判断一致；
6. 不允许由“最终接收标签”反向决定关键窗，选窗全过程不得使用正负标签；
7. 任一条件失败均 `augmentation_applied=false`，继续使用静态 `e0`，不是报错也不是强行拒识。

建议首轮搜索范围：

| 参数 | 候选值 |
|---|---|
| `PVAD_WIN_SEC` | 固定 1.0 |
| `PVAD_HOP_SEC` | 固定 0.2 |
| `PVAD_LAMBDA` | 0.05, 0.10, 0.20, 0.50 |
| `PVAD_SEED_MARGIN` | 0.00, 0.03, 0.05, 0.08 |
| `PVAD_TOP2_MARGIN` | 0.02, 0.04, 0.06 |
| `PVAD_MIN_CONSECUTIVE` | 1, 2, 3 |
| `PVAD_SOURCE` | raw-only, raw+se48k |
| `PRESENCE_BACKEND` | campplus, eres2netv2, vblink2_samresnet100 |

绝对阈值必须按编码器、语言和 `ENROLL_VAD` 设置分别校准，禁止把 ERes2NetV2 的 `locked_thr.json` 复用于 CAM++。

### 5.3 两阶段打分

为控制误接收，P0 不使用 `max(sim(eaug, all_windows))` 直接替代现有分数。采用两阶段方案：

1. **静态门控**：现有冻结策略先得到 `score_static`；
2. **灰区复核**：只有 `score_static` 位于 `[tau_gate-gray_low, tau_gate+gray_high]` 时才运行 ASE；
3. **增强复核**：用 `eaug` 对原 mix、分离流和关键窗周围的持续区域重新打分，使用稳健聚合值，例如连续窗 top-k 均值或分位数；
4. **决策约束**：增强仅能在校准得到的安全条件下救援或否决，不允许一个孤立的最大值接收负样本；
5. **原分数保留**：JSONL 同时写 `score_static`、`score_aug` 和最终有效分数，便于离线重放阈值而无需重跑分离/编码。

需要同时比较三种权限：

- `audit_only`：只记录增强分，不改变决策；
- `rescue_only`：仅救援静态门控附近的疑似正样本；
- `bidirectional_gray`：灰区内允许救援或否决。

首轮默认 `audit_only`，由离线 `optimize_gate_for_score.py` 选择是否开放决策权限。预期 `rescue_only` 有助于降低正样本误拒，但必须严查负样本右尾；`bidirectional_gray` 风险最高，不作为首个上线臂。

### 5.4 状态范围

竞赛离线评测默认：

- 每个 UID 从 `e0` 开始；
- 只允许在同一 CMD 内做 0 或 1 次自增强；
- 禁止按数据文件顺序把前一 UID 的 `eaug` 传给后一 UID；
- 分层、bootstrap 和阈值优化按用户或注册音频组划分，不能让同一注册人同时出现在训练折和验证折。

真实产品如果能提供稳定的匿名用户会话 ID，可以另开 `PVAD_SESSION_ADAPT=1` 实验：最多 5 次更新、TTL 30 分钟、始终锚定 `e0`、只用高置信接收片段、每次更新可回滚。会话适配不得用于当前独立样本的正式离线得分。

## 6. P1：完整三分类 PVAD

### 6.1 模型

新增一个与论文结构对齐、但具有清晰数据合同的模型：

- 输入：16 kHz mono；40 维 log-Mel，25 ms/10 ms；
- 条件：CAM++ 192 维注册声纹，或通过线性投影适配其他编码器维数；
- 主干：4 层 Conformer；
- 条件注入：每层或指定层的 FiLM；
- 输出：每帧 `P(NS), P(NTSS), P(TSS)`；
- 损失：三对 weighted pairwise loss，权重 `1/1/0.5`；
- 优化器基线：Adam，初始学习率 `5e-4`；上限 100 epoch，使用验证集早停；
- 输出决策不能只看单帧最大值，使用最短持续时间、迟滞和平滑后的 TSS 占比。

### 6.2 训练数据合同

每个训练样本必须记录目标说话人、其他说话人、噪声、混合增益、SNR、重叠区间及三类帧标签。混合类型至少覆盖：

- 目标单说话人；
- 非目标单说话人；
- 目标与一个/两个非目标人重叠；
- 仅噪声/静音；
- 目标 + 噪声；
- 非目标 + 噪声；
- 中英文短命令、口音、远场和 0–20 dB SNR。

注册段长度按 0.5、1.0、1.5 s 分桶，并模拟真实 KWS 的截断、首尾静音和增强失真。训练/验证/测试必须说话人互斥；DatasetA 正负标签只能用于最后的门控阈值与系统评价，不能被混入模型训练后再作为独立测试报告。

### 6.3 PVAD 到系统分数的映射

PVAD 输出的推荐特征包括：

```text
tss_ratio
tss_longest_run_ms
tss_p95
ntss_ratio
overlap_evidence
static_cosine
augmented_cosine
keyframe_margin
```

先用简单、可审计的逻辑回归或单调规则生成 `presence_score_pvad`，再在训练折优化阈值。禁止用复杂分类器在小规模 DatasetA 上追逐 in-sample 最优分。

## 7. 代码改造设计

建议新增文件：

```text
ve/scripts/pvad_self_augment.py          # P0 选窗、融合、稳健分数
ve/scripts/pvad_model.py                 # P1 Conformer+FiLM
ve/scripts/build_pvad_training_manifest.py
ve/scripts/train_pvad.py
ve/scripts/run_pvad_gate.py              # 批量推理和 JSONL
ve/scripts/merge_pvad_decisions.py       # 与现有决策合并
ve/scripts/optimize_pvad_gate.py         # 调用/复用官方分优化逻辑
ve/run_pvad_experiments.sh               # 阶段化实验入口
ve/download_pvad_models.sh               # 仅下载固定、可校验资产
ve/configs/pvad_ase.yaml
ve/tests/test_pvad_self_augment.py
ve/tests/test_pvad_state_isolation.py
ve/tests/test_pvad_coverage.py
```

现有文件的最小改动：

- `presence_encoder.py`：补充严格的 `embed_frames/embed_windows` 批量接口，禁止逐窗落临时 wav；
- `presence_gate.py`：接收可选 `augmented_embedding` 和 PVAD 审计字段，不改变默认路径；
- `run_extract.py`：增加 `--pvad-ase`、`--pvad-mode`、`--pvad-config`、`--pvad-cache-root`；打印命中/未命中/新算统计；
- `run_all.sh`：映射环境变量、检查配置标签匹配和 P0/P1 checkpoint；
- `optimize_gate_for_score.py`：把 `score_static/score_aug/pvad_features` 作为候选臂，继续使用相同 strict ASR 合同；
- `final_evaluate.py`：无需修改计分公式，只补充 PVAD 配置摘要和状态泄漏审计结果；
- `report_ve.py`：增加关键窗采用率、无可信窗率、回退率、按语言/噪声/叠话切片的 RR/FRR/CER/score。

### 7.1 JSONL 合同

每条 UID 至少新增：

```json
{
  "uid": "pos_0001",
  "pvad_mode": "ase_p0",
  "pvad_encoder": "campplus_zh",
  "pvad_config_sha256": "...",
  "enroll_embedding_sha256": "...",
  "augmentation_applied": true,
  "augmentation_reason": "trusted_keyframe",
  "keyframe": {
    "source": "raw_d1_spk1",
    "start_sec": 0.8,
    "end_sec": 1.8,
    "sim_e0": 0.51,
    "top2_margin": 0.07,
    "consecutive_support": 3
  },
  "lambda": 0.1,
  "score_static": 0.286,
  "score_aug": 0.334,
  "presence_score": 0.334,
  "decision": "accept",
  "fallback_used": false
}
```

不写完整声纹向量，避免泄露生物特征；若调试必须保存，使用受控目录、加密和自动过期策略。音频命名和已有 `raw/se48k`、`best/better` 约定保持一致。

### 7.2 缓存键

缓存键必须包含：

```text
sha256(input_wav_bytes)
sample_rate
encoder_name + encoder_weight_sha256
window_sec + hop_sec
VAD/preprocess_version
SE/separator_model_sha256
PVAD_config_sha256
```

任何字段变化都必须 cache miss。终端输出同一行进度，并在结束时显示：

```text
[PVAD_CACHE] total=1838 hit=... miss=... fresh=... invalid=... fallback=...
```

## 8. 系统化实验设计

### 8.1 固定不变项

- DatasetA manifest 和 UID；
- 全正样本 raw-best ASR 结果；若 ASR 音频或解码配置改变则全量重跑；
- `ASR_RESUME=0`、`STRICT_ENROLL=1`、`STRICT_EVAL=1`、`LIMIT=0`；
- 官方字符加权 CER；
- 冻结基线 `se48kgate_raw_best` 和 `locked.sep_route`；
- 门控臂不得通过标签选择关键窗，标签只用于训练折阈值优化和验证折计分。

### 8.2 分阶段筛选

**E0：一致性检查**

- `PVAD_ASE=0` 必须逐 UID 复现冻结基线决策；
- P0 audit-only 不改变音频和决策；
- 缓存开/关结果逐 UID 一致。

**E1：低成本 P0 影子打分**

- 编码器先测 CAM++（与论文一致）、ERes2NetV2（当前主线）和 VoxBlink2-100；
- 所有臂共享分离缓存和固定 ASR；
- 只输出 `score_aug`，用重复分层留出比较 `rescue_only` 与 `bidirectional_gray`；
- 按注册人/注册音频组、语言、正负标签分层，默认 30% holdout、至少 100 seeds；正式候选使用 500 seeds。

**E2：全量严格复核**

- 对入围的最多 2–3 个臂重新生成决策；
- 使用同一全正样本 raw-best ASR；
- `final_evaluate.py --strict`；
- 使用 `rank_final_candidates.py` 做 5,000 次配对分层 bootstrap。

**E3：独立注册集和鲁棒性**

- 至少三套 KWS 注册音频分别运行，阈值冻结，不允许在每套测试集重调；
- 噪声、重叠、语言、注册时长、性别/音色相近负样本切片；
- 重点观察非目标人声与注册人音色相近时的 FAR 右尾。

**E4：P1 完整 PVAD**

- 只有 E2/E3 证明 P0 稳定增益后启动；
- 先以静态注册声纹训练基线，再加一次自增强，最后才做会话长期更新；
- P1 仍需回到 E2/E3 的竞赛分合同，不能凭帧级 F1 上线。

### 8.3 多臂控制

大规模组合搜索容易把 DatasetA 过拟合。采用以下控制：

- 预注册参数网格和主指标；
- E1 最多保留 3 个臂，E2 最多保留 2 个臂；
- 以 holdout `score p05` 和相对基线 `delta p05` 排名，不取全量 oracle 阈值；
- 选中阈值后写入新的独立配置，例如 `locked_thr_pvad_ase_v1.json`，不得覆盖现有 `locked_thr.json`；
- 最终一次确认集只能评一次，失败后必须重新划分开发目标，不能继续针对确认集调参。

## 9. AutoDL 部署方案

### 9.1 目录

```text
/root/extract-main/                         # 本仓
/root/extract-main/ve/
/root/autodl-tmp/datasetA/
/root/autodl-tmp/<best_sep>/{pos,neg,index.jsonl}
/root/autodl-tmp/ve_models/
  eres2netv2_zh/
  campplus_zh/
  vblink2_samresnet100/
  pvad_ase/
    pvad_conformer_film.pt                  # 仅 P1
    model_manifest.json
/root/autodl-tmp/checkpoints/MossFormer2_ONNX/simple_model.onnx
/root/autodl-tmp/Qwen3-ASR-1.7B/
/root/autodl-tmp/ve_sep_cache/d1/
/root/autodl-tmp/ve_pvad_cache/
```

`model_manifest.json` 必须包含 checkpoint SHA256、训练配置 SHA256、git commit、输入采样率、特征版本、编码器名称和权重 SHA256、许可证/来源、导出时间。

### 9.2 环境安装

以下命令基于当前仓库已有脚本；P0 不需要额外训练框架：

```bash
cd /root/extract-main/ve
chmod +x ./*.sh
./setup_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ve
source .env_ve

ONLY=eres2netv2,campplus,vblink100 ./download_presence_encoders.sh
./download_moss_onnx.sh
./download_moss_se48k.sh
./download_qwen3_asr.sh
./check_env.sh

python - <<'PY'
import torch, modelscope, qwen_asr, onnxruntime
print('python dependencies ok')
print('cuda=', torch.cuda.is_available(), 'gpu=', torch.cuda.get_device_name(0))
PY
```

部署前把 `.env_ve` 中 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 设置为正整数，避免 `libgomp: Invalid value`。正式运行不依赖外网：模型下载完成后断网执行冒烟测试，验证所有模型均从本地加载。

P1 实现后需要在 `requirements.txt` 固定已验证的 PyTorch/torchaudio 兼容范围，不允许在线时无上限 `pip -U` 改变正式环境。推荐导出：

```bash
python -m pip freeze > /root/autodl-tmp/ve_models/pvad_ase/pip-freeze.txt
sha256sum /root/autodl-tmp/ve_models/pvad_ase/* > \
  /root/autodl-tmp/ve_models/pvad_ase/SHA256SUMS
```

### 9.2.1 上游参考仓库部署方式

只在需要复现实验或对齐数值时下载上游；生产推理不在运行时访问 GitHub：

```bash
cd /root/autodl-tmp
git clone https://github.com/asdhajksdh/ASE-PVAD.git ase-pvad-reference
cd ase-pvad-reference
git rev-parse HEAD | tee UPSTREAM_COMMIT
git status --short
```

下载后必须把实际 SHA 写入评测报告和 `third_party_manifest.json`。不要直接执行：

```bash
# 禁止在 ve 环境执行
conda env update -n ve -f environment.yml
```

若要核对上游训练脚本，使用隔离环境名称，并先清理 `environment.yml` 的绝对 prefix 和互相冲突的 CUDA 依赖：

```bash
# 仅研究复现示意；需先生成并评审 environment.autodl.yml
conda env create -n ase-pvad-ref -f environment.autodl.yml
conda activate ase-pvad-ref
```

正式 `extract-main` 推理仍使用 `ve` 环境。P1 模型应在隔离环境训练后导出纯 `state_dict` 或 TorchScript/ONNX，再由 `ve/scripts/run_pvad_gate.py` 加载；导出前后必须用固定输入做数值一致性测试。

### 9.3 P0 冒烟命令（实现后）

下面是本方案要求新增脚本后的目标命令，不应在合并实现前误认为当前仓库已经支持：

```bash
cd /root/extract-main/ve
source .env_ve
conda activate ve

BEST_SEP_DIR=/root/autodl-tmp/<best_sep> \
VE_OUT=/root/autodl-tmp/pvad_ase_smoke \
SEP_REUSE_ROOT=/root/autodl-tmp/ve_sep_cache \
PIPELINE=sep_route PRESENCE_BACKEND=campplus \
PVAD_ASE=1 PVAD_MODE=audit_only PVAD_CONFIG=./configs/pvad_ase.yaml \
CMD_SE=1 ENROLL_VAD=0 EXTRA_REJECT=0 \
STRICT_ENROLL=1 STRICT_EVAL=0 ASR_RESUME=0 \
LIMIT=32 SKIP_ASR=1 \
./run_all.sh 2>&1 | tee /root/autodl-tmp/pvad_ase_smoke.log
```

冒烟后必须检查：

```bash
python scripts/check_pvad_run.py \
  --ve-out /root/autodl-tmp/pvad_ase_smoke --expected 32 --strict
```

### 9.4 P0 全量筛选命令（实现后）

```bash
cd /root/extract-main/ve
source .env_ve
conda activate ve

EXP_ROOT=/root/autodl-tmp/ve_pvad_ase_v1 \
DATA_DIR=/root/autodl-tmp/datasetA \
KWS_CANDIDATES='strict_v2=/root/autodl-tmp/DatasetA_BINARY_MULTIMETRIC_GATE_STRICT_v2' \
PRESENCE_BACKENDS='campplus,eres2netv2,vblink2_samresnet100' \
PVAD_MODES='audit_only,rescue_only' \
PVAD_LAMBDAS='0.05,0.10,0.20,0.50' \
SEP_REUSE_ROOT=/root/autodl-tmp/ve_sep_cache \
HOLDOUT_FRAC=0.30 SEEDS=500 BOOTSTRAP_REPLICATES=5000 \
ASR_RESUME=0 STRICT_ENROLL=1 STRICT_EVAL=1 LIMIT=0 \
./run_pvad_experiments.sh 2>&1 | tee /root/autodl-tmp/log_pvad_ase_v1.txt
```

入围臂的最终复核：

```bash
python scripts/final_evaluate.py \
  --ve-out /root/autodl-tmp/ve_pvad_ase_v1/final/<arm> --strict

python scripts/rank_final_candidates.py \
  --candidate baseline=/root/autodl-tmp/ve_pvad_ase_v1/final/baseline \
  --candidate ase=/root/autodl-tmp/ve_pvad_ase_v1/final/<arm> \
  --replicates 5000 \
  --out-dir /root/autodl-tmp/ve_pvad_ase_v1/ranking
```

### 9.5 OOM 与故障降级

- 窗级 embedding 使用批量推理，先按总采样点数动态组 batch；
- 捕获 CUDA OOM 后清理当前 batch 引用和 CUDA cache，以 1/2 batch 重试，最小到 1；
- 单 UID 仍失败时写 `fallback_used=true`、完整错误类型和静态分数，继续该 UID 的冻结门控；
- 禁止因异常产生 `missing decision`；
- P1 checkpoint 缺失、哈希不符或编码器不匹配时启动即失败；只有明确设置 `PVAD_ALLOW_FALLBACK=1` 才允许全局回退；
- 进度条在同一行刷新，错误单独换行，结束后输出统一计数。

## 10. 验收标准

所有“必须”项通过才可进入下一阶段。

### 10.1 功能与数据合同

| 编号 | 验收项 | 必须标准 |
|---|---|---|
| F01 | 开关兼容 | `PVAD_ASE=0` 与当前冻结基线逐 UID 的分数、决策、extracted 路径一致；浮点允许误差 `1e-6` |
| F02 | 输入覆盖 | DatasetA manifest、decision、ASR 行数完整；UID 无重复；`coverage.errors=[]` |
| F03 | ASR 完整 | 所有正样本都有 ASR 行；接受样本最终 `status=ok`；空结果经过重试后为 0 |
| F04 | 状态隔离 | 独立 UID 改变遍历顺序后结果一致；无跨 UID 的 `eaug` 复用 |
| F05 | 缓存一致 | cache cold/warm 的分数和决策逐 UID 一致；错误缓存键不能命中 |
| F06 | 安全回退 | 无可信窗、单 UID OOM、SE 失败时均产生完整决策并回退静态门控 |
| F07 | 审计字段 | 每条均含配置/模型哈希、静态分、增强分、是否增强、关键窗来源和回退原因 |
| F08 | 生物特征保护 | 默认 JSONL 不保存原始 embedding；缓存和日志不泄露可逆声纹数据 |

### 10.2 数值与单元测试

| 编号 | 验收项 | 必须标准 |
|---|---|---|
| N01 | 向量归一化 | `e0/ek/eaug` 无 NaN/Inf，L2 norm 在 `[0.999,1.001]` |
| N02 | 确定性 | 固定输入、配置和权重重复 3 次，分数绝对差 `<1e-5`，决策完全一致 |
| N03 | 关键窗 | 人工构造 top1/top2、连续性、静音、短音频用例全部通过 |
| N04 | 漂移保护 | 最多 5 次更新后始终保留 `e0` 锚点；错误关键窗模拟不产生无界漂移 |
| N05 | 标签隔离 | 静态检查和测试证明推理/选窗代码不读取 `label/ref/text` 字段 |

### 10.3 模型离线指标（仅 P1）

| 编号 | 验收项 | 必须标准 |
|---|---|---|
| M01 | 说话人隔离 | 训练、验证、测试说话人集合交集为空 |
| M02 | 帧级收益 | 0.5–1.5 s 短注册测试上，TSS F1 相对静态 PVAD 提升至少 2 个绝对百分点 |
| M03 | 非目标保护 | NTSS AP 相对静态 PVAD 下降不超过 1 个绝对百分点 |
| M04 | 噪声/叠话 | 0–10 dB 和重叠子集的 TSS F1 均不低于静态 PVAD，至少一个提升 2 个百分点 |
| M05 | 校准 | 验证集与测试集的阈值/配置完全冻结，不在测试集重新选阈值 |

### 10.4 竞赛指标与上线 Go/No-Go

冻结上线候选基线为 `se48kgate_raw_best`：`RR=0.911392`、`CER=0.371215`、`score=0.770089`。若该结果在重新严格复跑时发生变化，以新产生且 coverage clean 的冻结报告为准，并记录原因。

| 编号 | 验收项 | Go 标准 |
|---|---|---|
| S01 | 正式计分 | `final_evaluate.py --strict` 成功，官方字符加权公式一致 |
| S02 | DatasetA 分数 | 候选全量 score 至少高于冻结基线 `+0.003` |
| S03 | 配对稳定性 | 5,000 次按标签/语言/注册人分层 bootstrap，`delta score p05 > 0` |
| S04 | RR 守门 | `RR_neg >= baseline_RR - 0.005` |
| S05 | CER 守门 | `CER_pos_micro <= baseline_CER + 0.005` |
| S06 | 语言切片 | zh、en 各自 score 均不低于基线超过 0.005；不得用总体增益掩盖单语言显著退化 |
| S07 | 风险切片 | 噪声、叠话、相似音色负样本的 FAR 绝对增幅均不超过 0.02 |
| S08 | 独立 KWS | 阈值不重调，在至少 3 套注册音频中，至少 2 套优于对应基线，且任何一套 score 退化不超过 0.003 |

如果 S02 未达到但 S03–S08 全通过，只能保留为实验臂；不能因结构新颖替换当前默认。若独立数据不足，结论必须写“DatasetA 候选”，不得写“可上线”。

### 10.5 AutoDL 性能与可运维性

| 编号 | 验收项 | 必须标准 |
|---|---|---|
| D01 | 离线启动 | 权重预下载后断网可完成 32 条冒烟 |
| D02 | 显存 | 门控阶段峰值显存 `<=12 GB`；RTX 4090 24 GB 全量无未恢复 OOM |
| D03 | P0 延迟 | 相对现有 extract 新增 p95 `<=100 ms/条`，门控总 p95 `<=350 ms/条` |
| D04 | P1 延迟 | 相对现有 extract 新增 p95 `<=180 ms/条`，门控总 p95 `<=450 ms/条` |
| D05 | 进度与统计 | 同行进度显示百分比/ETA；结尾有 cache、增强、回退、OOM 重试计数 |
| D06 | 可恢复 | 中断后重跑能按哈希安全复用，不重复覆盖有效结果，不混用旧 ASR |
| D07 | 可回滚 | `PVAD_ASE=0` 一键恢复冻结臂；回滚不需要重新分离或重新 ASR |
| D08 | 可复现 | 环境 freeze、git commit、模型/配置 SHA256 和完整命令随报告保存 |
| D09 | 上游隔离 | 上游仓库固定 commit；生产不依赖其网络、私有绝对路径或 `pvad` conda 环境 |
| D10 | 法务清晰 | P1 生产交付前取得明确许可证/授权，或证明模型和实现为合规的独立实现 |

## 11. 报告要求

每个正式臂的终端和 `summary.md/json` 至少输出：

- 负样本：总数、拒绝数、接收数、RR、FAR；
- 正样本：总数、接收数、误拒数、误拒率；
- 正样本最终字符加权 CER、总编辑错误数、总参考字符数；
- `score=(RR+1-CER)/2`；
- ASR ok/empty/error/retry/fallback 数；
- ASE 采用数、无可信窗数、静态回退数；
- 关键窗来源分布和 top2 margin 分布；
- cache hit/miss/invalid/fresh；
- 按 zh/en、噪声、叠话、注册时长的 RR/FRR/CER/score；
- 相对冻结基线的逐 UID 变化：`neg reject->accept`、`neg accept->reject`、`pos reject->accept`、`pos accept->reject`；
- paired bootstrap 的 mean/p05/p95；
- coverage errors、重复 UID 和模型/配置哈希。

报告中 PVAD F1、关键窗命中率和接受样本 CER 只能作为诊断项，不能替代正式竞赛分。

## 12. 风险清单与控制

| 风险 | 后果 | 控制措施 |
|---|---|---|
| 非目标人声被选为关键窗 | 错误增强后提高负样本误接收 | 绝对阈值 + top2 margin + 连续窗 + 跨流/跨编码器确认；默认 audit-only |
| 多流取最大值的多重比较偏差 | 流越多，负样本极值越高 | 不用孤立 max；用持续证据、稳健聚合、候选数校准和 strict rescue |
| SE 改变声纹或 ASR | RR/FRR 与 CER 同时漂移 | SE 只作为门控对照；ASR 默认 raw best；所有 ASR 音频变化单独重跑 |
| 自增强长期漂移 | 会话后段错误累积 | 始终锚定 `e0`、更新上限 5、TTL、回滚；离线独立样本禁用长期状态 |
| DatasetA 过拟合 | 公开/当前集高分，未知集下降 | 重复分层留出、候选限额、独立 KWS、阈值冻结、确认集只评一次 |
| 编码器阈值错配 | RR 看似固定或异常 | 配置写入 encoder/weights/enroll_vad/score_norm 标签，严格不匹配即失败 |
| ASR/decision 历史混用 | coverage error 或伪高分 | 正式 `ASR_RESUME=0`；按音频和解码配置哈希；strict coverage |
| AutoDL 外网或上游仓库变化 | 无法部署/结果漂移 | 固定上游 commit；所有生产代码进入 `extract-main`，权重本地化并校验，运行时不依赖 GitHub |
| 上游研究代码路径/环境不可移植 | 覆盖现有 ve 环境或读取错误数据 | 参考复现环境与生产环境隔离；移植最小结构；所有路径参数化并做 manifest 校验 |
| 上游无明确许可证或权重 | 法务风险/P1 无法直接推理 | P0 独立实现；P1 取得授权和权重或自行合规训练；未满足不得生产上线 |
| 资源超限 | OOM、吞吐过低 | 动态 batch、减半重试、缓存窗 embedding、静态门控回退 |
| 声纹隐私 | 生物特征泄露 | 默认不落 embedding；必要缓存加密、最小权限、TTL 和访问审计 |

## 13. 交付物与里程碑

### M0：设计冻结（1 天）

- 本文档评审通过；
- 参数网格、数据切分和基线报告 SHA256 冻结；
- 确认不同注册集可用范围和独立验证规则。

### M1：P0 影子模式（2–4 天）

- `pvad_self_augment.py`、缓存、JSONL、单元测试；
- CAM++/ERes2NetV2 批量窗 embedding；
- AutoDL 32 条断网冒烟；
- audit-only 全量报告。

### M2：P0 严格候选（2–3 天）

- rescue-only/bidirectional 对照；
- 500 seeds 阈值筛选和 5,000 bootstrap；
- 三套 KWS 与风险切片验证；
- 达标则生成 `locked_thr_pvad_ase_v1.json`，否则保持现有默认。

### M3：P1 训练与部署（1–3 周）

- 三分类混合数据生成器；
- Conformer+FiLM 训练、验证、导出；
- P1 与 P0/冻结基线同合同全量比较；
- checkpoint、manifest、freeze、SHA256 和 AutoDL 操作手册。

## 14. 最终上线判定

推荐的 Go/No-Go 顺序：

1. 功能、覆盖和状态隔离通过；
2. AutoDL 断网冒烟和全量无缺失；
3. DatasetA 严格分数及 paired p05 通过；
4. 独立 KWS 与噪声/叠话切片通过；
5. 性能、回退和隐私要求通过；
6. 只把通过全部标准的配置设为可选生产臂；经过一轮 shadow 监控后才改默认。

若任一步失败，保留 `se48kgate_raw_best` 或更早的冻结 `locked.sep_route`。论文方案的价值在于用 CMD 中的可信目标片段补足短注册声纹，而不是绕过严格评测；最终是否采用，只由完整覆盖下的 RR、字符加权 CER、竞赛分和独立稳定性决定。

## 参考资料

- Fuyuan Feng et al., [Adaptive Speaker Embedding Self-Augmentation for Personal Voice Activity Detection with Short Enrollment Speech](https://arxiv.org/abs/2601.12769), arXiv:2601.12769, accepted by ICASSP 2026.
- 用户提供的 [ASE-PVAD GitHub 仓库](https://github.com/asdhajksdh/ASE-PVAD)及其 [README](https://github.com/asdhajksdh/ASE-PVAD/blob/main/readme.md)（作为研究参考；固定 commit 后使用，不作为生产运行时网络依赖）。
- 论文原[匿名代码地址](https://anonymous.4open.science/r/ASE-PVAD-E5D6)（2026-08-26 检查为 HTTP 410）。
- 本仓现有说明：`ve/SYSTEMATIC_EXPERIMENTS.md`、`ve/GOAL_EXPERIMENTS.md`、`ve/SETUP.md`、`ve/VP/DESIGN.md`。
