"""Row: Web pages (HTML) / Mind2Web.

Raw HTML DOM observations. The `replace` treatment rewrites each element's
volatile backend_node_id into a stable content-derived handle, which is what
turns near-zero verbatim PIC into the family's best number.

Published: Prefix 0.71%  PIC 0.00%  PIC-proc 42.94%  in-sess 34.80  cross-sess 8.13
Treatment: replace. Data: HF osunlp/Mind2Web (streamed). Requires HF_TOKEN.
"""
import re
from collections import Counter

import _bootstrap
from pic_rules import lines_of
from prefix_disjoint import measure_disjoint
from analyze_mined import sessions_mind2web

IDATTR = re.compile(r'\s*backend_node_id="[^"]*"')


def h64(s):
    h = 1469598103934665603
    for c in s.encode():
        h = ((h ^ c) * 1099511628211) & (1 << 64) - 1
    return h


def replace_ids(html):
    occ = Counter()

    def sub(m):
        seg = m.group(0)
        cleaned = IDATTR.sub("", seg)
        occ[cleaned] += 1
        return re.sub(r'backend_node_id="[^"]*"',
                      f'backend_node_id="a{h64(cleaned + "|" + str(occ[cleaned] - 1)):016x}"',
                      seg)
    return re.sub(r"<[^>]*backend_node_id=\"[^\"]*\"[^>]*>", sub, html)


def measure(snaps_list, xform):
    u = [{"task": i, "snaps": [lines_of(xform(sn) if xform else sn) for sn in snaps]}
         for i, snaps in enumerate(snaps_list)]
    return measure_disjoint(u)


def main():
    m2 = sessions_mind2web(25)
    verbatim = measure(m2, None)
    replaced = measure(m2, replace_ids)
    _bootstrap.emit(
        "Web pages (HTML) / Mind2Web",
        {"prefix": 0.71, "pic": 0.00, "pic_proc": 42.94,
         "in_sess": 34.80, "cross_sess": 8.13},
        {"prefix_caching_pct": verbatim["prefix_caching_pct"],
         "pic_pct": verbatim["pic_pct"], "pic_proc_pct": replaced["pic_pct"],
         "in_session": replaced["in_session"], "cross_session": replaced["cross_session"]})


if __name__ == "__main__":
    main()
