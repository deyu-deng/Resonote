# Resonote

把一首歌（音频 / MIDI）或一份现成谱子，自动转成**能弹的指弹吉他谱**。

- **输入**：mp3/wav 音频、MIDI、Guitar Pro 3/4/5（.gp3/.gp4/.gp5/.gtp）、ASCII 文本谱（.txt/.tab）
- **输出**：`.gp5`（编辑）、`.musicxml`（标准交换）、MIDI、WAV 试听、HTML 预览、简易 PDF
- **核心思路**：好谱子是**减法**——分离出主旋律 + 低音支撑 + 少量和声点缀，而不是复刻整曲音频
- 测试：**206 个**（`pytest tests/`）——Windows 端实测全绿，mac 端待复跑

---

## 1. 快速开始

```bash
# 环境：mac 侧用 Python 3.11（basic_pitch 在 mac 绑 tensorflow-macos<2.15.1，见 §8）
# Windows 侧 3.12 已实测全绿
python -m venv .venv
uv pip install --python .venv -r requirements.txt   # 或 .venv 里的 pip install -r requirements.txt

# 解释器路径按平台不同：mac/linux = .venv/bin/python，Windows = .venv\Scripts\python.exe
# 下文一律记作 $PY
.venv/bin/python -m pytest tests/ -q               # Windows: .venv\Scripts\python.exe -m pytest tests/ -q

# 最小冒烟：内置样例旋律，不依赖音频模型
.venv/bin/python main.py --demo -o out.gp5 --html out.html
```

### 双机开发（mac + Windows 同一份 checkout）

git 是唯一同步通道，**任何东西都不要从一台拷到另一台**。下面这些本来就是各自一份、已被 gitignore：

| 资产 | 规则 |
|---|---|
| `.venv/` | 每台机器自己 `python -m venv` 建。**拷贝 venv 会带来断链解释器和整包重装问题**（曾把 mac 的 `.venv` 拷进 Windows 项目目录，直接跑不起来） |
| `.env` | 每台自己配；密钥不走 git |
| `fixtures/` | 版权资产，不走 git。mac 侧是指向云盘的软链，Windows 侧要自建同名结构；测得的 BPM/调性等**元数据**记在 `fixtures/manifest.json` |
| `runs/` | 产物按日期归档，不走 git |
| MuseScore / FluidSynth / node | 外部工具，路径按机器不同 → 用 `RESONOTE_MSCORE`、`RESONOTE_SOUNDFONT` 指过去，找不到时代码会静默降级并打印用的是哪个 |

常用命令：

```bash
# 音频 → 指弹谱（全链路：分离 → 转写 → 乐理 → 编曲 → 指法 → 导出）
# --pdf 装了 MuseScore 就走它出出版级排版，否则退回自写 writer（会打印用的是哪个）
$PY main.py "歌.mp3" -o out.gp5 --pdf out.pdf --html out.html \
    --midi out.mid --wav out.wav --musicxml out.musicxml

# 现成谱子（GP5 / 文本谱）→ 重新导出，**保留原作者指法，不重新编配**
$PY main.py song.gp5 -o mine.gp5 --pdf mine.pdf

# LLM 参与编曲判断（需 .env 配置，见 §4）
$PY main.py song.mid -o out.gp5 \
    --instruction "突出主旋律，和声点缀少而精，整体简单一些" --llm

# Web 界面
$PY web_server.py --port 8000
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
| L8 | 导出 GP5 / MusicXML / MIDI / WAV / PDF | 见下 | **pyguitarpro** (LGPL)、**pretty_midi** (MIT)、alphaTab (MPL-2.0)、**MuseScore Studio** (GPL，外部进程) |
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
$PY -m pytest tests/ -q                         # 206 个

# 2. 可弹性核验（同时音数 / 品位跨度——人手只有 4-5 品）
$PY experiments/verify_playability.py [产物.gp5]   # 默认取 runs/ 下最新

# 3. 独立实现交叉校验（不信 pyguitarpro 自己），gp5 / gp / musicxml / mid 通用
cd tools/alphatab && npm install              # 一次性，13MB，无传递依赖
node tools/alphatab/verify-score.mjs <产物文件>
```

**两个 reader 必须给出同一组数字。** `verify-score.mjs`（alphaTab）与
`verify_playability.py`（我们的 import 通路）从不同实现读同一份产物，若音数 /
同时音数 / 品位跨度不一致，说明其中一方的时间轴或延音链有问题——2026-09-27 就是
靠这个抓到 MusicXML 的 `<duration>` 与 `<type>` 不自洽（见 §6）。

**4. 排版出来的东西要用人眼看一遍。** 「PDF 生成了」不等于「谱是对的」：MuseScore
会把结构合法但语义错的输入渲染成空谱表或浮空数字（`staves=2` 现在就是这个状态）。
出图后转 PNG 看一眼第一页：

```bash
# $MS = 本机 MuseScore CLI。代码里已带 mac/Windows/Linux 候选路径，找不到就用
# RESONOTE_MSCORE 指过去；出图别写 /tmp，输出到当前目录即可
"$MS" -F -o preview.png <产物.gp5>
```

合格的《Melody》基线（分离开启）：1160 音 / 104 小节、同时最多 **4** 音（=拨弦上限）、
品位跨度平均 **3.8**、超 5 品占比 19%、0 空拍、0 同弦冲突；GP5 与 MusicXML 两条
通路读出的攻击数一致（1160）。

---

## 6. 已知坑（交接必读）

1. **GP5 一拍内同弦双音 = 文件损坏**（Guitar Pro 打不开）。修好后只剩**一道**防线：`arrange._resolve_string_collisions` 在唯一的出口处把冲突解决干净——能挪弦的挪，挪不了的（低于 5 弦空弦的音只有一根弦能按）直接丢，旋律优先于低音。`gp_export._emit_beat` 保留为最后一道保险，但**不能再依赖它**：以前"两道防线"的实际含义是两个导出器各按各的规则去重（GP5 留每拍第一个、MusicXML 的按弦字典留最后一个），于是同一份编配导出的 .gp5 与 .musicxml 会差一个半音，而两份各自看都自洽、可弹性校验也各报 0 冲突。新指法/导出逻辑不得破坏这条。
2. **`PlacedNote.pitch` 必须 = `OPEN_MIDI[string] + fret`**。两次栽在夹具音高不自洽上。
3. **`setuptools` 必须 <81**（basic_pitch 的 resampy 依赖 pkg_resources）。
4. **音频解码不要用 `torchaudio.load`**：2.9+ 已移到独立 torchcodec 包。`separate._load_audio()` 用 soundfile 优先，保留 torchaudio 作兜底。
5. **超出吉他音域的音**（分离后的贝斯低于 E2）：`_in_guitar_range()` 八度折叠。place_harmony 曾钳到 0 品静默破坏音高，已修。
6. **可弹性要按「正在发声」的音算**，不是同时起的音——延音低音照样占手。见 `enforce_playability`（注意：它 move 之后必须重算手位，曾因此死循环）。
7. **LLM 不可全信**：`_sanitize_edits()` 是确定性护栏（如 density=full 时抑制 drop harmony）。加新破坏性 op 要同步加护栏。
8. **密钥只在 `.env`**（已 gitignore）。提交前 `git diff --cached | grep -i key` 自查。
9. **`fixtures/` 整体 gitignore**：真实谱子有版权，绝不进公开仓库。
10. 沙箱/CI 环境：后台跑 basic_pitch 可能被杀（沙箱删除钩子无法确认），**长任务用前台**；exit 137 既可能是内存也可能是死循环，先查循环。
11. **MusicXML 一律「每弦一 voice」，不要回到单 voice + `<backup>`**。一根弦不可能同时发两音，所以每根弦就是一条线性时间线；单 voice 交织写法会被 alphaTab 直接拒绝（`Unsupported forward/backup detected`，实测 92 处），排版器也会错置。声部号**恒等于弦号**且全曲稳定——按「本小节第几个活跃弦」编号会让延音链跨声部断裂。
12. **`<duration>` 与 `<type>` 必须自洽**：5 个八分音符不是一个符值，得写成「二分 + 连八分」。读者按 `<type>` 排版、按 `<duration>` 计时，两者不一致时该声部后续全部漂移。小节线处同理——用 `<tie>`/`<tied>` 续写，别切断（切断=把持续音重新拨响）。

---

## 7. 目录说明

```
main.py           CLI 入口
pipeline.py       管线编排（run / run_import）
transcribe.py     L3 转写（MIDI / basic-pitch；YourMT3+ 已死路，见 §8 P5）
separate.py       L2 分离（demucs；soundfile 读音频）
analysis.py       L4 乐理
arrangement.py    L5 编曲 + LLM 判断层 + load_env_file
arrange.py        L6 指法 + enforce_playability
quantize.py       L7 量化
gp_export.py      L8 GP5 导出
musicxml_export.py L8 MusicXML 导出
midi_export.py    L8 MIDI/WAV
engraving.py      L8 调用 MuseScore 无头排版（出版级 PDF/SVG）
tab_pdf.py        L8 简易 PDF（没装排版器时的兜底）
preview.py        ASCII/HTML 预览
importers.py      GP3/4/5 + ASCII 文本谱导入
models.py         Note / PlacedNote / STANDARD_TUNING / tuning_labels
web_server.py     Web 后端
web/              前端（alphaTab 渲染）
tests/            206 个测试（test_m1..m17 编号按里程碑）
experiments/      OMR 原型 / GP7 转换 / 可弹性核验 / 节拍对照 bench_beats.py / 两进程分离流程
tools/alphatab/   alphaTab 独立校验器（node，npm install 后直接用，见 §5）
samples/          自造示例谱（可提交）
fixtures/         真实谱子测试集（gitignore，版权原因，仅本地）
runs/             每次跑出来的产物按日期归档（gitignore），仓库根目录只放源码
```

---

## 8. 当前状态与待办

**已完成里程碑**：M1-M16（详见 tests/ 编号）。最近：MusicXML 可被排版器读懂（`c98a4ce`）、声部合并（`354a083`）、**MuseScore 出版级排版接入**（`6be5189`）。

**本机已装好的外部工具**（都不经 brew：brew 6.0.9 比 tap/bottle 元数据旧，且 `github.com` 直连不通）：

| 工具 | 位置 | 用途 | 已验证 |
|---|---|---|---|
| MuseScore Studio 4.7.5 | `/Applications/MuseScore 4.app/Contents/MacOS/mscore` | 出版级 PDF/SVG/PNG，`--pdf` 自动使用 | ✅ 无头出图可用；`-s` 不存在；退出码不可信 |
| LilyPond 2.26.0 | `~/.local/bin/lilypond`（实体在 `~/.local/share/lilypond-2.26.0`） | 和弦框页 / Nashville 格谱 | ✅ `--pdf/--svg/--png` 可渲染 StaffGroup+TabStaff；`-b` 不支持 |
| FluidSynth 2.6.1 | `/opt/homebrew/bin/fluidsynth` | **真实吉他音色试听**，`--wav` 自动使用 | ✅ 主增益默认 0.1，必须 `-g 1.0` |
| SoundFont | `/Applications/MuseScore 4.app/Contents/Resources/sound/MS Basic.sf3` | FluidSynth 的音色库（本机原本一个都没有） | ✅ 用 `RESONOTE_SOUNDFONT` 可覆盖 |
| beat-this 1.1.0 | `.venv`（权重来自 cloud.cp.jku.at 而非 github） | **L4 拍点/重拍，已接入 `beats.py`** | ✅ 真实歌曲上把速度从错的 94 修正为 111 BPM；库自带示例把返回值顺序写反了 |
| muscriptor 0.3.0 | `.venv`（**权重 gated，未接入**） | L3 转写换代 | ❌ 权重需 HF 账号接受许可，见 §8 P5 |
| alphaTab 1.8.4 | `tools/alphatab/node_modules` | 独立校验器（不做印刷：无分页） | ✅ |

**brew 现在可用**（之前不行）：`~/.gitconfig` 里 `http.proxy=http://127.0.0.1:1087` 指向一个已经不存在的端口——代理改成了 TUN/VPN 模式（6 个 `utun` 接口，`github.com:443` 直连可达），git 却仍被要求走那个死端口，于是 `brew update` 和 `git push` 一起失败，而 curl 因为不读 git 配置所以一直好使。清掉即可：

```bash
git config --global --unset http.proxy && git config --global --unset https.proxy
```

在那之前用逐次覆盖绕过，不改全局配置：`GIT_CONFIG_PARAMETERS="'http.proxy=' 'https.proxy='" brew update`

**待办（按建议优先级）**：

| 优先级 | 事项 | 说明 |
|---|---|---|
| **P0** | **建 eval 集（真实音频已到位，缺权威谱）** | 音频侧已开：`fixtures/audio/taozhe-melody.mp3`（软链进云盘媒体库，不留第二份拷贝），`fixtures/manifest.json` 记着测得 111 BPM 及其三重证据。仍缺**人工核对过的谱**，所以音准/和声/指法三条只能自比、不能对答案。`experiments/bench_beats.py` 是现成的打分骨架 |
| P1 | ✅ 分离 | demucs 全链路已通（上游已归档，后路见 P5） |
| P2 | ✅ 出版级排版 | MuseScore 接好，`--pdf` 自动走它；自写 `tab_pdf` 降级为兜底 |
| P3 | 五线谱+六线谱同页 | `musicxml_export.build_musicxml(staves=2)` 编码已写但**MuseScore 渲成空五线谱 + 浮空数字**。多谱表 MusicXML 要用「写完 staff 1 → 整小节 `<backup>` → 写 staff 2」的布局，是另一件活 |
| P4 | ✅ 真音色试听 | FluidSynth 2.6.1 + MuseScore 自带的 MS Basic.sf3 已接入 `midi_export.render_preview`，`--wav` 自动使用，Karplus-Strong 降为兜底并会报告用的是哪个 |
| P5 | L3 换代（**卡在 HF 授权**） | `muscriptor` 0.3.0 已装进 `.venv`，但权重是 gated：① 浏览器登录并接受 <https://huggingface.co/MuScriptor/muscriptor-small>（免费、自动放行）② `uvx hf auth login` 或 `export HF_TOKEN=...`。做完这两步才能接线并评测。它自己也依赖 `beat-this>=1.1`，与 L4 的选择互相印证。分离后路：`msst` 0.1.0 在 PyPI |
| P6 | `place_bass` 选八度参考旋律把位 | 消掉剩余 17% 超 5 品段落 |
| P7 | 技法记号（H/P/击勾弦） | 纯规则可做；alphaTab 的 MusicXML 侧 bend/slide/hammer 都支持 |
| P8 | 单乐器改走 f0 | `rmvpe-onnx` 0.2.3 在 PyPI（MIT）。人声/贝斯近单音，f0 比把 basic_pitch 套在 stem 上准得多 |

**已核实为死路 / 不要碰**：madmom（PyPI 停在 2018，3.10+ 装不上，许可证暧昧）、YourMT3+（`yourmt3-plus` 在 PyPI 404，上游仓库 2024-11 冻结）、MR-MT3（无官方实现）、TuxGuitar（无头 CLI 不存在，PDF 只在 GUI 里）、alphaTab 做印刷（**没有分页**，官方文档自己写着 "no strict print-page display yet"）、abjad 3.31 与 librosa 1.0（都要 Python ≥3.12，我们是 3.11）。

**未解决**：`transcribe.py` 的 YourMT3 后端（38 处引用、被 test_m1 断言）与 `analysis.py` 的 madmom 后端（34 处引用）都是**不可能被满足的扩展点**——前者上游冻结且 PyPI 无此包，后者装不上 3.10+。它们目前能优雅失败，所以没在其它改动里顺手删；清理需要单独一次，连测试一起改。另：整曲混音转写 F1≈0.43（2026 年流行多轨的客观基线只有 onset F1≈29%，所以天花板在 L3 不在编曲）；Web 无鉴权；**downbeat 已检测到但还没被用上**——`beats.py` 现在从波形取回拍点和重拍（真实歌曲上把速度从错的 94 修正为 111 BPM），但小节线仍锚在第一个音上，`analysis._build_bars` 与两个导出器的 anchor 都还没读 `analysis.downbeats`。


---

## 9. 定位（前负责人拍板，勿轻易改）

对外发行的产品 · 开源 · 死磕**全自动直出** · 「能弹」= 音准对 + 节奏密度合理 + 指法舒服 + 和声对，**四条全要** · 架构本地优先。
