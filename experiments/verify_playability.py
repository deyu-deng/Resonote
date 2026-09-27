"""独立核验一份 GP5 产物的可弹性 — 不依赖管线自己的说法。

用法:  python experiments/verify_playability.py [产物.gp5]
不给参数时核验 runs/ 下最近生成的那份 .gp5。
"""
import glob
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from importers import load_tab


def _newest_gp5() -> str:
    candidates = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "*.gp5")),
                        key=os.path.getmtime)
    if not candidates:
        raise SystemExit("no .gp5 under runs/ — give a path as argv[1]")
    return candidates[-1]


path = sys.argv[1] if len(sys.argv) > 1 else _newest_gp5()
print(f"核验: {os.path.relpath(path, ROOT)}")
s = load_tab(path)
placed = s.placed
print('GP5 回读: %d 音, %d 小节, tempo %.1f' % (len(placed), s.measures, s.tempo))

# 同一时刻(量化到 1/16 拍)最多同时按几个音
step = 60.0 / s.tempo / 4.0
buckets = defaultdict(list)
for p in placed:
    buckets[round(p.onset / step) * step].append(p)

counts = [len(v) for v in buckets.values()]
print('同一时刻音数分布:', dict(sorted(Counter(counts).items())))
print('最大同时音数:', max(counts))

# 品位跨度(可弹性: 人手 4-5 品)
spans = []
for v in buckets.values():
    if len(v) >= 2:
        f = [p.fret for p in v]
        spans.append(max(f) - min(f))
if spans:
    print('同时按弦品位跨度: 平均 %.1f, 最大 %d, 超5品的拍占比 %.0f%%'
          % (sum(spans) / len(spans), max(spans),
             100 * sum(1 for x in spans if x > 5) / len(spans)))

# 品位分布(是否在高把位扎堆)
hi = sum(1 for p in placed if p.fret >= 12)
print('12品以上占比: %.0f%% (%d/%d)' % (100 * hi / len(placed), hi, len(placed)))
