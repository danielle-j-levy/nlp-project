"""Row: Web pages (live websites) / NNetNav-Live.

Accessibility-tree observations from live sites. Same `replace` treatment as the
WebArena row (volatile [123] handles -> stable content handles).

Published: Prefix 20.31%  PIC 11.82%  PIC-proc 14.79%  in-sess 11.79  cross-sess 3.00
Treatment: replace. Data: HF stanfordnlp/nnetnav-live (train.jsonl). Requires HF_TOKEN.
"""
import re
from collections import Counter, defaultdict

import _bootstrap
from pic_rules import lines_of
from prefix_disjoint import measure_disjoint
import analyze_mined
from huggingface_hub import hf_hub_download

BR = re.compile(r"\[(\d+)\]")


def h64(s):
    h = 1469598103934665603
    for c in s.encode():
        h = ((h ^ c) * 1099511628211) & (1 << 64) - 1
    return h


def handle(snapshot):
    occ = Counter()
    out = []
    for line in snapshot.split("\n"):
        cleaned = BR.sub("[.]", line)
        occ[cleaned] += 1
        o = occ[cleaned] - 1
        i = [0]

        def sub(m):
            aid = h64(f"{cleaned}|{o}|{i[0]}")
            i[0] += 1
            return f"[a{aid:016x}]"
        out.append(BR.sub(sub, line))
    return "\n".join(out)


def load_live():
    p = hf_hub_download("stanfordnlp/nnetnav-live", "train.jsonl", repo_type="dataset")
    by = defaultdict(list)
    for i, line in enumerate(open(p)):
        if i >= 4000:
            break
        import json
        r = json.loads(line)
        m = analyze_mined.OBS.search(r.get("prompt") or "")
        if m:
            by[r.get("id")].append(m.group(1))
    return [s for s in by.values() if len(s) >= 2][:40]


def measure(snaps_list, xform):
    u = [{"task": i, "snaps": [lines_of(xform(sn) if xform else sn) for sn in snaps]}
         for i, snaps in enumerate(snaps_list)]
    return measure_disjoint(u)


def main():
    live = load_live()
    verbatim = measure(live, None)
    handles = measure(live, handle)
    _bootstrap.emit(
        "Web pages (live websites) / NNetNav-Live",
        {"prefix": 20.31, "pic": 11.82, "pic_proc": 14.79,
         "in_sess": 11.79, "cross_sess": 3.00},
        {"prefix_caching_pct": verbatim["prefix_caching_pct"],
         "pic_pct": verbatim["pic_pct"], "pic_proc_pct": handles["pic_pct"],
         "in_session": handles["in_session"], "cross_session": handles["cross_session"]})


if __name__ == "__main__":
    main()
