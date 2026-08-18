# Resonote 升级计划与目标架构

> 编写者：Senior Developer
> 日期：2026-07-11（2026-07-11 更新：LLM 升级为 L5 判断层；新增"极致 UX"体验层）
> 配套文档：`RESEARCH.md`（调研依据）、`overview.md`（交付概述）

---

## 一、产品目标（北极星）

**拖入任意音乐文件（mp3/wav/mid），一键生成"上手就能弹"的指弹独奏吉他谱（GP5 + MusicXML + MIDI + 交互预览）。**

"上手就能弹"的三条硬标准：
1. **可演奏性**：指法符合人手工程学，无不可能把位/手型
2. **音乐性**：旋律在高位、低音在低位、和声填充合理、voice leading 平滑
3. **完整编曲**：不是单旋律，而是 melody + bass + 和声的独奏改编

### 体验北极星（本次新增，最高优先级）
**极致的用户体验**不是"能上传就行"。目标是：拖入文件的那一刻起，用户处于一个**电影级、可感知进度、可交互、可对话**的编曲工作流里。
- 拖入即反馈（磁性玻璃上传区 + 文件名/时长/预估难度）
- **管线剧场**：9 层处理以可视化的方式逐层点亮，用户"看见 AI 在编曲"
- **交互式谱面**：6 线谱在线预览，难度/风格实时切换，播放头动画
- **自然语言编辑**："副歌低音再饱满点" → 直接改编排，无需重跑全管线
- 光/暗/跟随系统三态主题，60fps，加载 < 1.5s，全端响应式

---

## 二、目标架构（9 层管线 + 体验层）

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 9  体验层 ⭐ Premium Web（拖入→管线剧场→交互谱面→NL编辑）│
│           Laravel/Livewire/FluxUI + 高级 CSS + Three.js       │
├─────────────────────────────────────────────────────────────┤
│  Layer 8  导出层  GP5 | MusicXML | MIDI | 交互 HTML(VexFlow) │
├─────────────────────────────────────────────────────────────┤
│  Layer 7  量化层  节拍网格量化 + 节奏修正                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 6  指法层  MIDI-to-Tab(BART) 主 / 增强 DP 回退        │
├─────────────────────────────────────────────────────────────┤
│  Layer 5  编曲层  ⭐规则骨架 + LLM 判断层（双引擎，必选）     │
│            角色分配 + chord voicing + voice leading + 音乐性复核│
├─────────────────────────────────────────────────────────────┤
│  Layer 4  乐理层  和弦识别 + 节拍/BPM + 调性 + 段落          │
├─────────────────────────────────────────────────────────────┤
│  Layer 3  转录层  YourMT3+(主) / Basic-Pitch(回退) / MIDI直读│
├─────────────────────────────────────────────────────────────┤
│  Layer 2  分离层  demucs HTDemucs v4 / Mel-RoFormer(人声捷径)│
├─────────────────────────────────────────────────────────────┤
│  Layer 1  预处理  格式归一化 + 重采样 + 单声道               │
└─────────────────────────────────────────────────────────────┘
```

### 与现有代码的映射

| 现有模块 | 升级后归属 | 动作 |
|---------|-----------|------|
| `transcribe.py` | Layer 2+3 | 拆分为 `separate.py` + `transcribe.py`，换 YourMT3+ |
| `arrange.py` | Layer 5+6 | 拆为 `arrange.py`(编曲) + `fingering.py`(指法)，新增编曲层 |
| `gp_export.py` | Layer 7+8 | 拆为 `quantize.py` + `export.py`，新增 MusicXML/MIDI |
| `preview.py` | Layer 8/9 | 重写为 VexFlow 交互预览 + 接入体验层 |
| `models.py` | 全局 | 扩展数据模型（角色、和弦、节拍、LLM 建议） |
| `main.py` | Layer 9 | 保留 CLI，新增 `web/`（Laravel/Livewire）体验层 |

---

## 三、核心创新层详解：Layer 5 编曲引擎（规则骨架 + LLM 判断层）

这是现有管线完全缺失、也是产品差异化的关键。把多声部塞进一把吉他的"音乐判断"。

**双引擎架构（已确认：LLM 为必选判断层，非可选增强）：**
```
规则引擎（确定性、可测、可回退）
   │  产出候选编曲：角色分配 + voicing + voice leading
   ▼
LLM 判断层（音乐性复核 + 取舍 + 风格/段落编排）
   │  在候选上做"像人一样"的判断，可整体否决/局部调整
   ▼
最终编排（规则兜底：LLM 失败时回退到规则产物）
```
- LLM 失败 / 超时 / 降级 → 自动回退规则引擎产物，绝不整段崩。
- 规则引擎保证"一定能弹"，LLM 把"能弹"推到"好听 + 像人编的"。

### 5.1 角色分配（Role Assignment，规则引擎）
- `melody`：主旋律（人声轨优先，否则最高声部）→ 高音区（1-3 弦）
- `bass`：和弦根音/低音线 → 低音区（5-6 弦）
- `harmony`：和弦内音填充 → 中音区（3-4 弦）
- `optional`：装饰音/过门，可省略降难度

### 5.2 Chord Voicing 库（规则引擎）
- 开放和弦（C/Am/F）、横按（F/Bm）、指弹特有 voicing（CAGED + bass-on-low-string）

### 5.3 Voice Leading（规则引擎）
- 公共音保持、声部移动最小化、避免交叉

### 5.4 难度控制（产品差异化）
- `easy`：仅 melody+bass；`medium`：+基础 harmony；`hard`：完整+装饰音

### 5.5 LLM 判断层（必选，从 M3 起设计）⭐
规则骨架产出后，LLM 在候选上做"人味"判断（ChatMusician / ComposerX 思路）：
- **砍音取舍**：知道 9 音可省、3 音不能省（规则只能按"重要性评分"硬排，常砍错色彩音）
- **段落编排**：verse 简化 / chorus 饱满 / 间奏炫技（需对歌曲结构的理解）
- **风格切换**：民谣指弹 / 古典 chord-melody / 爵士 walking bass（一句 prompt 切风格）
- **复杂和弦**：转调、离调、延伸和弦的把位取舍
- **自然语言编辑接口**：承接体验层 L9 的对话式修改

---

## 四、数据模型扩展（`models.py`）

```python
@dataclass
class Note:
    pitch: int
    onset: float
    duration: float
    velocity: int = 100
    instrument: str = ""   # 来自 YourMT3+
    track_id: int = 0

@dataclass
class AnalyzedNote(Note):
    role: str = ""         # melody/bass/harmony/optional
    chord: str = ""
    beat: float = 0.0

@dataclass
class PlacedNote:
    pitch: int
    onset: float
    duration: float
    string: int
    fret: int
    finger: int
    pluck: str = ""
    role: str = ""
    effect: str = ""       # hammer/pull/slide/bend
    llm_note: str = ""     # 新增：LLM 对该音的取舍理由（可解释性）
```

---

## 五、里程碑计划（UX 前置，LLM 必选）

> 体验层原型（L9 设计壳）本轮即产出，用于给团队定调；生产集成在 M6。

### M1 — 地基重建（源分离 + AMT 升级）
- [ ] 新增 `separate.py`：封装 demucs，输出 stems
- [ ] `transcribe.py` 接入 YourMT3+，Basic-Pitch 降级回退
- [ ] `models.py` 加 instrument/track_id
- **验收**：`python main.py song.mp3` 跑通分离→多轨 Note，比原 Basic-Pitch 明显准

### M2 — 乐理分析层
- [ ] 新增 `analysis.py`：madmom 和弦 + 节拍 + 调性 + 段落
- [ ] 主旋律轨选择策略（人声优先 → 最高声部）
- **验收**：流行歌输出正确和弦进行 + BPM

### M3 — 编曲引擎（核心）⭐ 规则骨架 + LLM 判断层
- [ ] `arrange.py`（重写）：角色分配 + voicing + voice leading（规则骨架）
- [ ] **LLM 判断层接口同步设计**：候选 → 复核 → 回退协议；prompt 工程 + 结构化输出解析
- [ ] 指弹 voicing 库（CAGED 开放把位起步）
- [ ] 难度参数 `--difficulty`
- **验收**：C-G-Am-F 旋律 MIDI → 输出含 bass+melody+和声的 PlacedNote；LLM 在候选上给出可解释的取舍建议且可回退

### M4 — 指法生成升级
- [ ] `fingering.py`：接入 MIDI-to-Tab BART；保留增强 DP 回退
- [ ] 加入 barre/滑音/推弦 effect 标注
- **验收**：对比 DP 与 BART 可演奏性（人工抽检 5 段）

### M5 — 量化与导出
- [ ] `quantize.py`：节拍网格量化
- [ ] `export.py`：GP5 + MusicXML(music21) + MIDI
- **验收**：浏览器/CLI 拿到 GP5 + MusicXML + MIDI 三件套

### M6 — 体验层（极致 UX，生产集成）⭐
- [ ] Laravel/Livewire/FluxUI 体验层：拖入 → 管线剧场 → 交互谱面
- [ ] VexFlow 交互预览 + 播放头 + 难度/风格实时切换
- [ ] 自然语言编辑接口（对接 L5 LLM 判断层）
- [ ] 光/暗/系统三态主题、磁性交互、60fps、< 1.5s 首屏
- **验收**：拖入 mp3，管线剧场可视化跑完，拿到可交互谱面 + NV 编辑可用

### M7 — LLM 生产化与成本
- [ ] 多 Agent 编排（ComposerX 思路）、缓存、降级策略、成本护栏
- [ ] 用户研究：LLM 建议采纳率 > 50%
- **验收**：3 首样例 LLM 调整被采纳率达标，单曲成本可控

---

## 六、技术决策记录（ADR 摘要）

1. **ADR-001 GP5 为主输出**：兼容性最广（Guitar Pro/TuxGuitar/MuseScore）。
2. **ADR-002 编曲引擎 = 规则骨架 + LLM 判断层（双引擎，必选）**：LLM 做"音乐性判断"，规则做"可演奏性兜底"。LLM 不稳定 → 规则回退，绝不整段崩。（**已更新：LLM 从 M6 可选升级为 L5 必选判断层**）
3. **ADR-003 YourMT3+ 主 / Basic-Pitch 回退**：精度差 2 倍；YourMT3+ 需 GPU，CPU 走 Basic-Pitch 兜底。
4. **ADR-004 不自训指法模型，先接预训练**：MIDI-to-Tab 开源权重优先；不可用则增强 DP 交付。
5. **ADR-005 demucs 4-stem 起步**：6-stem 钢琴质量差且重；吉他从 "other" 二次识别。
6. **ADR-006 体验层用 Laravel/Livewire/FluxUI + 高级 CSS + Three.js**：极致 UX 的既定技术栈；原型阶段可用静态 HTML/CSS/JS + Canvas 验证体验，生产迁移 Livewire。

---

## 七、给团队的代码质量规范

1. 类型注解强制（`mypy --strict`）
2. 分层测试 + 端到端 `tests/test_e2e.py`
3. 中间产物 `--dump-intermediate` 可观测
4. 配置外置 `config.yaml`
5. 依赖分层：核心层零 ML 依赖；ML 层 lazy import
6. 可回退：每个 ML 环节必须有非 ML 回退路径

---

## 八、下一步行动

**本轮已交付**：
- `RESEARCH.md` 调研报告
- `overview.md` 概述
- **`web/` 极致 UX 原型**（拖入 → 管线剧场 → 交互谱面 → NL 编辑设计壳，可本地跑）

**下阶段双线并行**：
- 后端：开 M1（`separate.py` + YourMT3+）
- 前端：M6 体验层向 Livewire/FluxUI 迁移，接真实管线

团队先 review `web/` 原型定体验调性，同时我启动 M1 后端地基。
