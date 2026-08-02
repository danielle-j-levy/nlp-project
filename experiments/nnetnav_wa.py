"""Row: Web pages (a11y tree) / NNetNav-WA (WebArena).

Accessibility-tree observations. PIC is measured verbatim and after the
`replace` treatment, which rewrites the volatile element handles ([123]) into
stable content-derived handles so identical widgets hash alike across steps.
PIC-proc / in-sess / cross-sess are the treated numbers.

Published: Prefix 37.32%  PIC 4.50%  PIC-proc 10.03%  in-sess 5.62  cross-sess 4.41
Treatment: replace. Data: HF stanfordnlp/nnetnav-wa. Requires HF_TOKEN.
"""
import re
from collections import Counter

import _bootstrap
from pic_rules import lines_of
from prefix_disjoint import measure_disjoint
from analyze_mined import sessions_nnetnav

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


def measure(snaps_list, xform):
    u = [{"task": i, "snaps": [lines_of(xform(sn) if xform else sn) for sn in snaps]}
         for i, snaps in enumerate(snaps_list)]
    return measure_disjoint(u)


def main():
    wa = sessions_nnetnav(40)
    verbatim = measure(wa, None)
    handles = measure(wa, handle)
    _bootstrap.emit(
        "Web pages (a11y tree) / NNetNav-WA (WebArena)",
        {"prefix": 37.32, "pic": 4.50, "pic_proc": 10.03,
         "in_sess": 5.62, "cross_sess": 4.41},
        {"prefix_caching_pct": verbatim["prefix_caching_pct"],
         "pic_pct": verbatim["pic_pct"], "pic_proc_pct": handles["pic_pct"],
         "in_session": handles["in_session"], "cross_session": handles["cross_session"]})


if __name__ == "__main__":
    main()
