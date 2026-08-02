"""Row: Retrieved documents / tau-Knowledge.

Banking knowledge base. Verbatim, each task concatenates its required documents
into one prompt (PIC). The `align` treatment packs the KB into fixed >=500-token
buckets so a document reused across tasks always lands on the same span grid;
PIC-proc / in-sess / cross-sess are the bucket-aligned numbers.

Published: Prefix 34.77%  PIC 4.32%  PIC-proc 47.74%  in-sess 0.00  cross-sess 47.74
Treatment: align. Data: local data/tauknowledge/banking_knowledge/
"""
import json

import _bootstrap
from pic_rules import lines_of, nt, MIN
from prefix_disjoint import measure_disjoint
from rag500 import tau_docs


def main():
    docs = tau_docs()
    tasks = json.load(open("data/tauknowledge/banking_knowledge/tasks.json"))
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks") or list(tasks.values())
    sel = [{"task": t.get("id"), "req": r} for t in tasks
           if (r := [d for d in sorted(t.get("required_documents") or []) if d in docs])]

    concatenated = measure_disjoint(
        [{"task": t["task"], "snaps": [[u for d in t["req"] for u in lines_of(docs[d])]]}
         for t in sel])

    buckets, buf, n = [], [], 0
    for d in sorted(docs):
        buf.append(d)
        n += nt(docs[d])
        if n >= MIN:
            buckets.append(buf)
            buf = []
            n = 0
    if buf:
        buckets[-1].extend(buf) if buckets else buckets.append(buf)
    owner = {d: i for i, b in enumerate(buckets) for d in b}
    aligned = measure_disjoint(
        [{"task": t["task"],
          "snaps": [[u for d in buckets[i] for u in lines_of(docs[d])]
                    for i in sorted({owner[d] for d in t["req"]})]} for t in sel])

    _bootstrap.emit(
        "Retrieved documents / tau-Knowledge",
        {"prefix": 34.77, "pic": 4.32, "pic_proc": 47.74,
         "in_sess": 0.00, "cross_sess": 47.74},
        {"prefix_caching_pct": concatenated["prefix_caching_pct"],
         "pic_pct": concatenated["pic_pct"], "pic_proc_pct": aligned["pic_pct"],
         "in_session": aligned["in_session"], "cross_session": aligned["cross_session"]})


if __name__ == "__main__":
    main()
