# Resonote 开源生态与前沿技术调研报告

> 调研日期：2026-07-11
> 调研者：Senior Developer
> 目的：为"音频/ MIDI → 指弹独奏吉他谱"产品升级，找齐可复用/整合的开源项目与 LLM 前沿方案

---

## 一、现有代码底子（基线评估）

| 模块 | 现状 | 等级 | 瓶颈 |
|------|------|------|------|
| `transcribe.py` | Basic-Pitch (F1≈0.43) + pretty_midi | ⚠️ 偏弱 | 对混音歌曲几乎不可用；无源分离；单声道旋律假设 |
| `arrange.py` | Sayegh 风格 DP + 局部手型窗口 | ⚠️ 偏弱 | 只做(弦,品,指)分配，**不做编曲**；无 bass/melody/和声角色；无 barre/滑音 |
| `gp_export.py` | pyguitarpro 写 GP5 | ✅ 可用 | 仅 GP5；量化是简单 2 的幂 snap；无 MusicXML/PDF |
| `preview.py` | ASCII + 简陋 HTML | ✅ 可用 | 无交互、无音频对齐 |
| `main.py` | CLI | ✅ 可用 | 无 Web UI、无拖拽 |

**根本性局限**：现有管线把"编曲"问题降维成了"单旋律指法映射"。但指弹独奏谱的本质是**把多声部（旋律+低音+和声填充）塞进一把吉他**——这恰恰是当前缺失的核心层。

---

## 二、开源项目深度调研（按管线分层）

### Layer 1 — 音频源分离（Source Separation）

混音歌曲必须先分离出吉他/人声 stem，否则 AMT 精度灾难性下降。这是当前管线完全缺失的环节。

| 项目 | 模型 | 吉他 SDR | 协议 | 评价 |
|------|------|---------|------|------|
| **demucs (HTDemucs v4)** | Hybrid Transformer U-Net | ~7.2 (4-stem) / 吉他 stem 可用 | MIT | Meta 官方，生态最成熟，MUSDB-HQ SDR 9.0；6-stem 版可直出吉他轨 |
| **BS-RoFormer / Mel-RoFormer** | Band-split + RoPE Transformer | 7.5+ (吉他) | MIT (lucidrains 实现) | 2024 SOTA，频带分离更精细；mvsep 实测吉他 SDR 7.51 |
| **Mel-RoFormer (vocal+melody)** | 同上，二合一微调 | — | 开源 | **分离+旋律转录一步到位**，省掉后续 AMT 一步，对人声主导歌曲特别香 |
| MDX-Net / Spleeter | 老模型 | 较低 | — | 已过时，不推荐 |

**选型结论**：
- **默认**：`demucs` (HTDemucs v4) —— 稳定、生态好、6-stem 能直出吉他
- **高精度路径**（可选）：`BS-RoFormer`（ZFTurbo/Music-Source-Separation-Training 权重）
- **人声旋律捷径**：Mel-RoFormer 二合一模型，跳过 AMT 直接出主旋律 MIDI

---

### Layer 2 — 自动音乐转录（AMT）

| 项目 | F1 (Slakh2100) | 多乐器 | 协议 | 评价 |
|------|---------------|--------|------|------|
| Basic-Pitch (Spotify) | 0.43 | 是 | Apache-2.0 | 轻量(ONNX)但精度垫底；适合快速原型 |
| MT3 (Google) | 0.57 | 是 | Apache-2.0 | T5 架构，2021 SOTA，有乐器泄漏问题 |
| MR-MT3 | ~0.6 | 是 | 开源 | MT3 改进，加记忆保持缓解乐器泄漏 |
| **YourMT3+** | **0.8456** | 是 | MIT | **2024-07 SOTA**；PyTorch；hierarchical attention + MoE；跨数据集 stem 增强；**支持人声直转**（免分离） |
| PerceiverTF | 0.819 | 是 | 开源 | YourMT3+ 同期，次优 |

**选型结论**：
- **主力**：`YourMT3+` —— 精度比 Basic-Pitch 翻倍，多乐器，PyTorch 易部署
- **回退**：Basic-Pitch（轻量，无 GPU 环境兜底）
- **MIDI 输入**：直接 `pretty_midi`，不变

---

### Layer 3 — 旋律提取与音高追踪（针对人声主导歌曲）

| 工具 | 用途 | 协议 | 评价 |
|------|------|------|------|
| **CREPE** | 单音高追踪 | MIT | 独奏乐器误差 <2.1%；多声部会受和声干扰 |
| SPICE | 歌声音高 | Apache-2.0 | Google，singer-friendly |
| Melodia (vamp) | 旋律线提取 | GPL | 老牌，需 vamp 插件，集成麻烦 |
| Mel-RoFormer melody | 分离+转录二合一 | 开源 | **推荐**：一步到位 |

**选型结论**：人声主导歌曲用 Mel-RoFormer 二合一；器乐曲走 demucs + YourMT3+。

---

### Layer 4 — 指法生成（Tablature Inference）⭐ 核心层

这是现有 `arrange.py` 所在层，也是重复造轮子风险最高的地方。

| 方案 | 方法 | 数据 | 协议 | 评价 |
|------|------|------|------|------|
| 现有 arrange.py | Sayegh DP + 手型窗口 | 无 | — | 最简基线 |
| natecdr/MIDI2Tabs (tuttut) | HMM + Viterbi | 无 | MIT | ~80⭐，最成熟的开源指板映射 |
| jgollub1/guitar_dp | DP 最优指法 | 无 | MIT | 与本项目思路几乎一致，可参考代价函数 |
| **MIDI-to-Tab (arXiv 2408.05024)** | **BART encoder-decoder + MLM** | **DadaGP (26K 谱)** | 开源 | **2024-08 SOTA**；用户研究证明显著优于 DP；预训练+微调 |
| From MIDI to Rich Tablatures (2407.09052) | 多属性约束优化 + 风格统计 | mySongBook | 开源 | 加入 articulation/expressive 技巧（滑音、推弦等） |
| MusicGuitarTab/GuitarTab | TabCNN + 后处理 | DadaGP | 开源 | 后处理把精度拉到 99.92% |
| robust-guitar-tabs/code | TabCNN + 音色鲁棒 | EGFXSET | 开源 | 处理效果器/音色变化 |

**DadaGP 数据集**（关键资产）：
- 26,181 首 Guitar Pro 文件，739 流派
- 含指法、演奏技巧、多乐器
- tokenized 格式，适合序列模型
- 地址：github.com/dada-bots/dadagp

**选型结论**：
- **主路径**：接入 `MIDI-to-Tab` 预训练 BART 模型（或自训），DadaGP 知识注入
- **回退**：保留并增强 DP（参考 jgollub1/guitar_dp 的代价函数：hand stretch / barre / position shift）
- **增强**：参考 "From MIDI to Rich Tablatures" 加入 articulation 标注

---

### Layer 5 — 和弦 / 节拍 / 调性识别（乐理层，全新）

指弹编曲必须知道和弦进行，才能决定 bass 音与和声填充。当前管线完全缺失。

| 工具 | 能力 | 协议 | 评价 |
|------|------|------|------|
| **madmom** | 和弦 + 节拍 + downbeat + onset，CRF+CNN | BSD-3 | MIR 工业标准，MIREX 2024 baseline；易集成 |
| Essentia | 和弦 + 大量音频特征 | AGPLv3 | C++ 高性能，但协议需注意 |
| chord-detection (python) | 简单和弦检测 | MIT | 轻量回退 |
| librosa | chroma + 节拍 | ISC | 通用底座 |

**选型结论**：`madmom` 做和弦+节拍主力，`librosa` 做 chroma 特征底座。

---

### Layer 6 — LLM 在乐理与编曲中的应用（前沿增强）

LLM 不替代 DSP，而是做"音乐性判断"——这是规则引擎最难 coded 的部分。

| 项目 | 方法 | 能力 | 评价 |
|------|------|------|------|
| **ChatMusician** (ACL 2024) | LLaMA2 + ABC notation 持续预训练 | 乐理理解、多声部生成、和弦条件生成 | MusicTheoryBench 超越 GPT-3.5；开源 |
| **ComposerX** (2024-04) | 多 Agent (GPT-4) | 复调作曲、和声约束 | 证明 multi-agent 显著提升质量 |
| MuPT | ABC notation 预训练 | 符号音乐生成 | 提出 SMT-ABC 多轨同步表示 |
| Qwen2-Audio 等 | 多模态 LLM | 音频→乐谱 | 探索性，精度不及专用 AMT |

**LLM 适合接入的点**（不替代核心管线）：
1. **段落结构标注**（verse/chorus/bridge）→ 影响编曲密度
2. **和声重配建议**（次要功能，给用户多版本）
3. **指弹编曲的"音乐性"判断**：哪些音值得做 melody、哪些做填充
4. **自然语言交互编辑**："把副歌难度降低"、"低音再饱满点"

**选型结论**：M6 可选里程碑，用 ChatMusician 思路做乐理 Agent；不作为 MVP 依赖。

---

### Layer 7 — 导出格式生态

| 格式 | 库 | 现状 | 建议 |
|------|----|----|------|
| GP5 | pyguitarpro | ✅ 已用 | 保留，兼容性最好 |
| GP7/GPX | — | ❌ pyguitarpro 不支持 | 不做（需逆向） |
| **MusicXML** | music21 / pymusicxml | ❌ 缺失 | **新增**：跨格式交换标准，可转 PDF/ MIDI/ 各种谱 |
| MIDI | pretty_midi | ❌ 缺失 | 新增：导出改编后 MIDI 供 DAW 二次编辑 |
| PDF | MusicXML→MuseScore/LilyPond | ❌ 缺失 | 新增：打印用 |
| 交互 HTML | 自研 | ✅ 简陋 | 升级：VexFlow / Tablature.js 渲染 + 音频对齐 |

**选型结论**：用 `music21` 同时产出 MusicXML + MIDI；HTML 预览用 VexFlow 重写。

---

### Layer 8 — 同类竞品对标

| 产品 | 能力 | 对标价值 |
|------|------|---------|
| **Fingerstyle Tab MCP Server** (blooper20) | demucs + Basic-Pitch + 角色映射 + 和弦 + 自动转调 + Claude MCP | **几乎同款**，架构可直接参考 |
| audio2guitar | 6 阶段管线（分离→多音高→onset→和弦→beat→段落），$6.99/月 | 商业参考，验证管线合理性 |
| Guitar2Tabs (Klangio) | 支持 GP/MIDI/MusicXML/PDF 导出 | 导出格式参考 |
| Moises | 只分离+和弦，不出 tab | 反面教材：用户要 fret 级 |

---

## 三、技术选型总表

| 管线层 | 推荐方案 | 回退方案 | 新增依赖 |
|--------|---------|---------|---------|
| 源分离 | demucs HTDemucs v4 | — | demucs (MIT) |
| AMT 转录 | YourMT3+ | Basic-Pitch | yourmt3 (MIT) |
| 旋律提取 | Mel-RoFormer 二合一 | CREPE | bs-roformer |
| 和弦/节拍 | madmom | librosa chroma | madmom (BSD) |
| 指弹编曲 | 自研规则引擎 + (可选)LLM | — | (可选) ChatMusician |
| 指法生成 | MIDI-to-Tab (BART) | 增强 DP | dadagp 权重 |
| 量化 | 节拍网格量化 | 现 2 的幂 snap | madmom beat |
| 导出 | GP5 + MusicXML + MIDI | — | music21 (BSD) |
| 预览 | VexFlow 交互 HTML | 现 ASCII | — |
| UI | CLI + Web 拖拽 (FastAPI) | CLI | fastapi, uvicorn |

---

## 四、关键风险与边界

1. **YourMT3+ 部署成本**：PyTorch 模型，需 GPU 加速才实用；CPU 可跑但慢。需提供 Basic-Pitch 回退。
2. **DadaGP 权重**：MIDI-to-Tab 是否放出预训练权重需确认；若无，需自训（DadaGP 数据可申请）。
3. **demucs 资源**：6-stem 模型较重，4-stem 够用；首跑需下载 ~80MB 权重。
4. **madmom 协议**：BSD-3 友好；Essentia 是 AGPLv3，避免商用。
5. **指弹编曲是半结构化问题**：规则引擎能覆盖 70%，剩下 30% 的"音乐性"判断是 LLM 的甜区，但 LLM 不是 MVP 依赖。
6. **多乐器输入处理**：YourMT3+ 输出多轨，需决定"主旋律"是谁（人声优先？最高声部？），这是产品决策点。

---

## 五、一句话结论

> 现有管线缺了"源分离 + 乐理分析 + 指弹编曲"三大层，且 AMT 和指法生成两个环节都有 2024 SOTA 开源方案可直接替换。**升级路径清晰，无技术黑盒**，关键在工程整合而非算法攻关。
