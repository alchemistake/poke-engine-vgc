#!/usr/bin/env python3
"""Group a valgrind DHAT profile by the engine source line that allocated.

  python3 bench/dhat_sites.py target/vg/dhat.out
"""
import collections
import json
import re
import sys

d = json.load(open(sys.argv[1]))
ft = d['ftbl']
sites = collections.Counter()
for pp in d['pps']:
    locs = []
    for i in pp['fs']:
        m = re.search(r': ([\w:<>{}, ]+?) \((battle|events|state|damage|rng|bench_doubles|io)\.rs:(\d+)\)', ft[i])
        if m:
            locs.append(f'{m.group(1).split("::")[-1][:30]}@{m.group(2)}:{m.group(3)}')
            if len(locs) == 2:
                break
    sites[' < '.join(locs) or '?'] += pp['tbk']
total = sum(sites.values())
print(f'{total} allocations')
for k, v in sites.most_common(30):
    print(f'{v:8} {v / total * 100:5.1f}% {k}')
