"""Row: Operational telemetry / ITBench SRE.

SRE tool outputs are JSON envelopes {"output": ..., "metadata": ...}. The
`relocate` treatment moves the volatile metadata into a sidecar (still counted
in the denominator) so the stable output body can form reusable spans.
PIC-proc / in-sess / cross-sess are the relocated numbers.

Published: Prefix 0.18%  PIC 0.00%  PIC-proc 7.71%  in-sess 7.71  cross-sess 0.00
Treatment: relocate. Data: HF ibm-research/ITBench-Trajectories. Requires HF_TOKEN.
"""
import json
import re

import _bootstrap
from huggingface_hub import HfApi, hf_hub_download
from pic_rules import lines_of, nt
from prefix_disjoint import measure_disjoint

TASK = re.compile(r"(Scenario-\d+)")
REPO = "ibm-research/ITBench-Trajectories"


def load():
    files = sorted(s.rfilename for s in HfApi().dataset_info(REPO).siblings
                   if s.rfilename.endswith("session.jsonl"))
    sess = []
    for rf in files:
        p = hf_hub_download(REPO, rf, repo_type="dataset")
        m = TASK.search(rf)
        task = m.group(1) if m else rf
        raw_s, rel_s, side = [], [], 0
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            pl = r.get("payload") or {}
            if pl.get("type") != "function_call_output":
                continue
            o = pl.get("output") or ""
            raw = o
            try:
                inner = json.loads(o)
                if isinstance(inner, dict) and isinstance(inner.get("output"), str) \
                        and inner.get("metadata") is not None:
                    o = inner["output"]
                    side += nt(json.dumps(inner["metadata"], ensure_ascii=False))
                else:
                    o = json.dumps(inner, ensure_ascii=False)
                    raw = o
            except (ValueError, TypeError):
                pass
            if len(o) >= 80:
                rel_s.append(lines_of(o))
                raw_s.append(lines_of(raw))
        if len(rel_s) >= 2:
            sess.append({"task": task, "raw": raw_s, "rel": rel_s, "side": side})
        if len(sess) >= 40:
            break
    seen = set()
    return [s for s in sess if not (s["task"] in seen or seen.add(s["task"]))]


def main():
    dd = load()
    verbatim = measure_disjoint([{"task": s["task"], "snaps": s["raw"]} for s in dd])
    relocated = measure_disjoint([{"task": s["task"], "snaps": s["rel"]} for s in dd],
                                 sidecar=sum(s["side"] for s in dd))
    _bootstrap.emit(
        "Operational telemetry / ITBench SRE",
        {"prefix": 0.18, "pic": 0.00, "pic_proc": 7.71,
         "in_sess": 7.71, "cross_sess": 0.00},
        {"prefix_caching_pct": verbatim["prefix_caching_pct"],
         "pic_pct": verbatim["pic_pct"], "pic_proc_pct": relocated["pic_pct"],
         "in_session": relocated["in_session"], "cross_session": relocated["cross_session"]})


if __name__ == "__main__":
    main()
