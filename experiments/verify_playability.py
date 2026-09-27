"""独立核验 out.melody.gp5 的可弹性 — 不依赖管线自己的说法。"""
import sys
sys.path.insert(0, '/Users/ciel/Projects/Resonote')

from collections import defaultdict
from importers import load_tab

s = load_tab('/Users/ciel/Projects/Resonote/out.melody.gp5')
placed = s.placed
print('GP5 回读: %d 音, %d 小节, tempo %.1f' % (len(placed), s.measures, s.tempo))

# 同一时刻(量化到 1/16 拍)最多同时按几个音
step = 60.0 / s.tempo / 4.0
buckets = defaultdict(list)
for p in placed:
    buckets[round(p.onset / step) * step].append(p)

counts = [len(v) for v in buckets.values()]
from collections import Counter
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
