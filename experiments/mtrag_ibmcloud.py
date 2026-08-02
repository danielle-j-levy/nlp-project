"""Row: Retrieved documents / MT-RAG ibmcloud.

MT-RAG human conversations; each retrieval turn's passages are one snapshot.
Conversations are grouped by collection and this row reports the ibmcloud
collection. Treatment: none.

Published: Prefix 0.44%  PIC 13.76%  PIC-proc 13.76%  in-sess 12.86  cross-sess 0.90
Data: local data/mtrag/conversations/conversations_human.json
"""
from collections import defaultdict

import _bootstrap
from prefix_disjoint import measure_disjoint
from rag500 import sessions_mtrag


def main():
    bycol = defaultdict(list)
    for s in sessions_mtrag("human"):
        bycol[s["task"].split("/")[0]].append(s)
    target = None
    for col, ss in sorted(bycol.items()):
        if "ibmcloud" in col:
            target = (col, ss)
    if target is None:
        raise SystemExit("no ibmcloud collection found in MT-RAG human conversations")
    col, ss = target
    d = measure_disjoint([{"task": s["task"], "snaps": s["snaps"]} for s in ss])
    print(f"MT-RAG collection: {col} ({len(ss)} conversations)", flush=True)
    _bootstrap.emit("Retrieved documents / MT-RAG ibmcloud",
                    {"prefix": 0.44, "pic": 13.76, "pic_proc": 13.76,
                     "in_sess": 12.86, "cross_sess": 0.90}, d)


if __name__ == "__main__":
    main()
