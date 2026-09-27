# Resonote

把一首歌（音频 / MIDI）或一份现成谱子，自动转成**能弹的指弹吉他谱**。

- **输入**：mp3/wav 音频、MIDI、Guitar Pro 3/4/5（.gp3/.gp4/.gp5/.gtp）、ASCII 文本谱（.txt/.tab）
- **输出**：`.gp5`（编辑）、`.musicxml`（标准交换）、MIDI、WAV 试听、HTML 预览、简易 PDF
- **核心思路**：好谱子是**减法**——分离出主旋律 + 低音支撑 + 少量和声点缀，而不是复刻整曲音频
- 测试：**176 个**（`pytest tests/`）

---

## 1. 快速开始

```bash
# 环境：Python 3.11（不要用 3.13，basic_pitch 在 macOS 绑死 tensorflow-macos<2.15.1）
python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt   # 或 pip install -r requirements.txt

# 跑测试（无需任何模型/密钥）
.venv/bin/python -m pytest tests/ -q

# 最小冒烟：内置样例旋律，不依赖音频模型
.venv/bin/python main.py --demo -o out.gp5 --html out.html
```

常用命令：

```bash
# 音频 → 指弹谱（全链路：分离 → 转写 → 乐理 → 编曲 → 指法 → 导出）
.venv/bin/python main.py "歌.mp3" -o out.gp5 --pdf out.pdf --html out.html \
    --midi out.mid --wav out.wav --musicxml out.musicxml

# 现成谱子（GP5 / 文本谱）→ 重新导出，**保留原作者指法，不重新编配**
.venv/bin/python main.py song.gp5 -o mine.gp5 --pdf mine.pdf

# LLM 参与编曲判断（需 .env 配置，见 §4）
.venv/bin/python main.py song.mid -o out.gp5 \
    --instruction "突出主旋律，和声点缀少而精，整体简单一些" --llm

# Web 界面
.venv/bin/python web_server.py --port 8000
```

---

## 2. 架构：9 层管线

`pipeline.py` 是唯一编排入口（CLI 与 web 共用）：

| 层 | 职责 | 模块 | 依赖的开源库 |
|---|---|---|---|
| L2 | 音源分离（人声/贝斯/其他/鼓） | `separate.py` | **demucs** (MIT) |
| L3 | 音频→音符（AMT 转写） | `transcribe.py` | **basic-pitch** (Apache-2.0)；可选 YourMT3+ |
| L4 | 乐理：节拍/和弦/调性/段落 | `analysis.py` | librosa (ISC) |
| L5 | 编曲引擎：角色分配/voicing/**LLM 判断层** | `arrangement.py` | 自研 + LLM（MiniMax，OpenAI 兼容协议） |
| L6 | 指法：DP 选(弦,品) + 左手指法 + 防同弦冲突 | `arrange.py` | 自研 |
| L7 | 量化：onset/duration 吸附到节拍网格 | `quantize.py` | 自研 |
| L8 | 导出 GP5 / MusicXML / MIDI / WAV / PDF | 见下 | **pyguitarpro** (LGPL)、**pretty_midi** (MIT)、alphaTab (MPL-2.0) |
| L9 | Web UI（stdlib http.server + 静态页） | `web_server.py` + `web/` | alphaTab 渲染 |

**设计原则：能用成熟开源就不自研。** 自研的只有三块：L4 乐理、L5/L6 编曲与指法（产品差异化核心，无成熟开源方案）、各格式的导出 writer。

---

## 3. 输入 / 输出格式

### 输入

| 格式 | 支持 | 说明 |
|---|---|---|
| `.mid/.midi` | ✅ | pretty_midi 直读 |
| `.mp3/.wav` | ✅ | basic-pitch 转写；开分离后走 demucs |
| `.gp3/.gp4/.gp5/.gtp` | ✅ | **保留原作者指法与调弦**，跳过 L1-L7 不重新编配 |
| `.txt/.tab`（ASCII 谱） | ✅ | 自动识别调弦（含 DADGAD/Drop D）、两种行序、BOM/CRLF |
| 图片谱（OMR） | 🔶 原型 | `experiments/omr/`，谱线检测已 100%，符号识别 ~80% |
| MusicXML / `.gpx` / `.gp` | ❌ | 未写导入（`.gp` 可经 alphaTab 从 GP5 转出，见 `experiments/gp5_to_gp7.mjs`） |

**注意**：GP5 ≠ 过时。Guitar Pro 8 原生向下兼容打开 .gp3/4/5，GP5 是 tab 生态兼容面最广的格式；需要 GP7/8 的 `.gp` 时用 alphaTab 的 `Gp7Exporter` 转换（已验证）。

### 输出

| 格式 | 用途 | 生成器 |
|---|---|---|
| `.gp5` | 编辑（Guitar Pro/TuxGuitar/MuseScore 全兼容） | pyguitarpro |
| `.musicxml` | 标准交换，出版级渲染用 MuseScore 打开 | 自写（语法按官方教程逐条核对，alphaTab 独立验证过） |
| `.pdf` | 打印对照（零依赖兜底） | 自写 writer；装 MuseScore 后可用 `mscore out.musicxml -o out.pdf` 得出版级 |
| `.mid` / `.wav` | 试听 | pretty_midi / Karplus-Strong 合成 |
| `.html` | 浏览器预览（alphaTab 渲染 + 试听） | 自写 |

调弦全链路保留：DADGAD、Drop D 等非标准调弦从输入贯穿到 GP5/MusicXML/PDF。

---

## 4. LLM 判断层

L5 编曲引擎有一个判断层，规则版永远可用，LLM 版按指令做出**可解释**的编曲决策。

**配置**（项目根 `.env`，已 gitignore，绝不提交）：

```
RESONOTE_LLM_API_KEY=...
RESONOTE_LLM_BASE_URL=https://api.minimaxi.com/v1
RESONOTE_LLM_MODEL=MiniMax-Text-01
```

当前 provider 是 MiniMax。两个坑：
- 端点必须 `api.minimaxi.com/v1`（`.io` 是国际站，CN key 会 401）
- 不支持 `response_format=json_object`，且返回常包 ```json 围栏 → 代码已适配

**行为**：`--instruction` 非空且配置了 key → 自动走 LLM；`--llm` 强制；否则走规则层。LLM 挂掉/返回坏 JSON 时**不崩**，回退规则层并在判断日志写明。

**查看 LLM 的决策**：加 `--dump-intermediate DIR`，读 `DIR/arrangement.json` 的 `judgment_log`，每条 edit 都带理由，例如：

```
judged by LLM layer
LLM edit: set_density all = light（用户希望整体编配简单一些…）
LLM edit: drop harmony = 1.06（为了突出主旋律，和声应点缀少而精…）
```

> 历史坑：`.env` 曾从未被加载（`is_llm_configured()` 读的是进程环境），导致 LLM 层长期静默失效。`load_env_file()` 已在两个入口调用，**别删**。

---

## 5. 验证方法学（这个项目最重要的纪律）

**「文件生成了」≠「能弹」。** 每次交付前必须验证：

```bash
# 1. 全量测试
.venv/bin/python -m pytest tests/ -q          # 176 个

# 2. 可弹性核验（同时音数 / 品位跨度——人手只有 4-5 品）
.venv/bin/python experiments/verify_playability.py

# 3. 独立实现交叉校验（不信 pyguitarpro 自己）
cd /Users/ciel/.workbuddy/binaries/node/workspace && \
  node verify_alphatab.mjs <产物.gp5>        # alphaTab（MPL-2.0）独立解析
```

合格的《Melody》基线（分离开启）：同时最多 **4** 音（=拨弦上限）、品位跨度平均 **3.8**、超 5 品占比 19%、0 空拍、0 同弦冲突。

---

## 6. 已知坑（交接必读）

1. **GP5 一拍内同弦双音 = 文件损坏**（Guitar Pro 打不开）。两道防线：`arrange._resolve_string_collisions` + `gp_export._emit_beat`。新指法/导出逻辑不得破坏。
2. **`PlacedNote.pitch` 必须 = `OPEN_MIDI[string] + fret`**。两次栽在夹具音高不自洽上。
3. **`setuptools` 必须 <81**（basic_pitch 的 resampy 依赖 pkg_resources）。
4. **音频解码不要用 `torchaudio.load`**：2.9+ 已移到独立 torchcodec 包。`separate._load_audio()` 用 soundfile 优先，保留 torchaudio 作兜底。
5. **超出吉他音域的音**（分离后的贝斯低于 E2）：`_in_guitar_range()` 八度折叠。place_harmony 曾钳到 0 品静默破坏音高，已修。
6. **可弹性要按「正在发声」的音算**，不是同时起的音——延音低音照样占手。见 `enforce_playability`（注意：它 move 之后必须重算手位，曾因此死循环）。
7. **LLM 不可全信**：`_sanitize_edits()` 是确定性护栏（如 density=full 时抑制 drop harmony）。加新破坏性 op 要同步加护栏。
8. **密钥只在 `.env`**（已 gitignore）。提交前 `git diff --cached | grep -i key` 自查。
9. **`fixtures/` 整体 gitignore**：真实谱子有版权，绝不进公开仓库。
10. 沙箱/CI 环境：后台跑 basic_pitch 可能被杀（沙箱删除钩子无法确认），**长任务用前台**；exit 137 既可能是内存也可能是死循环，先查循环。

---

## 7. 目录说明

```
main.py           CLI 入口
pipeline.py       管线编排（run / run_import）
transcribe.py     L3 转写（MIDI / basic-pitch / YourMT3+）
separate.py       L2 分离（demucs；soundfile 读音频）
analysis.py       L4 乐理
arrangement.py    L5 编曲 + LLM 判断层 + load_env_file
arrange.py        L6 指法 + enforce_playability
quantize.py       L7 量化
gp_export.py      L8 GP5 导出
musicxml_export.py L8 MusicXML 导出
midi_export.py    L8 MIDI/WAV
tab_pdf.py        L8 简易 PDF（兜底）
preview.py        ASCII/HTML 预览
importers.py      GP3/4/5 + ASCII 文本谱导入
models.py         Note / PlacedNote / STANDARD_TUNING / tuning_labels
web_server.py     Web 后端
web/              前端（alphaTab 渲染）
tests/            176 个测试（test_m1..m15 编号按里程碑）
experiments/      OMR 原型 / GP7 转换 / 可弹性核验 / 两进程分离流程
samples/          自造示例谱（可提交）
fixtures/         真实谱子测试集（gitignore，版权原因，仅本地）
```

---

## 8. 当前状态与待办

**已完成里程碑**：M1-M15（详见 tests/ 编号）。最近：分离管线全链路跑通（`5079b32`）、MusicXML 导出（`43af5f7`）、GP5→GP7 转换（`d6ca245`）、LLM .env 修复（`a60edce`）、数字对弦渲染修复（`2d5ffce`）。

**待办（按建议优先级）**：

| 优先级 | 事项 | 说明 |
|---|---|---|
| P0 | 建 eval 集 | 「音频 + 权威谱」配对 2-3 首，量化四条 done（音准/节奏/指法/和声）。没有它一切优化无度量 |
| P1 | ✅ 分离 | torchaudio 已装，demucs 全链路已通 |
| P2 | 换 FluidSynth 真音色 | `brew install fluidsynth` + SoundFont，替换 Karplus-Strong，听感质变 |
| P3 | madmom 换 L4 节拍/和弦 | 现在和弦从音符反推（弱）；madmom 是 SOTA |
| P4 | `place_bass` 选八度参考旋律把位 | 消掉剩余 19% 超 5 品段落 |
| P5 | 技法记号（H/P/击勾弦） | 纯规则可做：同弦相邻音 + 时序重叠 → hammer-on |
| P6 | MuseScore 集成 | 检测到 `mscore` 则自动出出版级 PDF |
| P7 | OMR 符号识别 | 模板匹配/小模型，节奏在符干+横梁里 |

**未解决**：整曲混音转写 F1≈0.43（clean 单乐器才好）；Web 无鉴权；`style.css` vs `styles.css` 疑似重复待清理。

---

## 9. 定位（前负责人拍板，勿轻易改）

对外发行的产品 · 开源 · 死磕**全自动直出** · 「能弹」= 音准对 + 节奏密度合理 + 指法舒服 + 和声对，**四条全要** · 架构本地优先。
