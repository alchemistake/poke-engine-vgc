#!/usr/bin/env python3
"""Summarize a samply profile: self and inclusive time per function, or the
callers of one function.

  samply record --save-only --unstable-presymbolicate -o prof.json.gz -r 4000 \
      ./target/release/bench-doubles bench/games.json 200
  python3 bench/profile_top.py prof.json.gz                  # top functions
  python3 bench/profile_top.py prof.json.gz --callers malloc  # who calls it
"""
import bisect
import collections
import gzip
import json
import re
import sys


def load(path):
    p = json.load(gzip.open(path))
    s = json.load(open(path.replace('.json.gz', '.json.syms.json')))
    strings = s['string_table']
    libsyms = {}
    for d in s['data']:
        tab = sorted(d['symbol_table'], key=lambda e: e['rva'])
        libsyms[d['debug_name']] = ([e['rva'] for e in tab], tab)
    t = p['threads'][0]
    ft, fu, res = t['frameTable'], t['funcTable'], t['resourceTable']
    cache = {}

    def name(fi):
        if fi in cache:
            return cache[fi]
        f = ft['func'][fi]
        r = fu['resource'][f]
        if r is None or r < 0:
            n = t['stringArray'][fu['name'][f]]
        else:
            lib = p['libs'][res['lib'][r]]['debugName']
            if lib not in libsyms:
                n = lib
            else:
                rvas, tab = libsyms[lib]
                i = bisect.bisect_right(rvas, ft['address'][fi]) - 1
                n = strings[tab[i]['symbol']] if i >= 0 else '?'
        n = re.sub(r'::h[0-9a-f]{16}$', '', n)
        cache[fi] = n
        return n

    return t, name


def main():
    path = sys.argv[1]
    t, name = load(path)
    stk, smp = t['stackTable'], t['samples']
    stacks = [si for si in smp['stack'] if si is not None]
    if '--callers' in sys.argv:
        target = sys.argv[sys.argv.index('--callers') + 1]
        depth = 3
        par = collections.Counter()
        for si in stacks:
            if target in name(stk['frame'][si]):
                chain, p = [], stk['prefix'][si]
                while p is not None and len(chain) < depth:
                    chain.append(name(stk['frame'][p]).split('::')[-1][:40])
                    p = stk['prefix'][p]
                par[' < '.join(chain)] += 1
        for k, v in par.most_common(15):
            print(v, k)
        return
    selfc, incl = collections.Counter(), collections.Counter()
    for si in stacks:
        first, seen = True, set()
        while si is not None:
            n = name(stk['frame'][si])
            if first:
                selfc[n] += 1
                first = False
            if n not in seen:
                incl[n] += 1
                seen.add(n)
            si = stk['prefix'][si]
    tot = len(stacks)
    print('samples', tot)
    print('--- self')
    for n, c in selfc.most_common(30):
        print(f'{c / tot * 100:5.1f}% {n[:110]}')
    print('--- inclusive')
    for n, c in incl.most_common(40):
        print(f'{c / tot * 100:5.1f}% {n[:110]}')


main()
