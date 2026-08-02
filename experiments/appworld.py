"""Row: Stateful API responses / AppWorld.

Tool outputs are API responses whose only cross-run-stable content is tiny. The
`mask` treatment (diagnostic) blanks a volatile JSON field (release_date) to show
the ceiling; PIC-proc / in-sess / cross-sess are the masked numbers.

Published: Prefix 16.93%  PIC 0.21%  PIC-proc 0.40%  in-sess 0.00  cross-sess 0.40
Treatment: mask. Data: local data/appworld/hf_traces/halo_gemini3flash_traces.jsonl
"""
import json
import re

import _bootstrap
from pic_rules import lines_of
from prefix_disjoint import measure_disjoint
from analyze_mined import apply_spec

MSG = re.compile(r"llm\.input_messages\.(\d+)\.message\.role")
TRACES = "data/appworld/hf_traces/halo_gemini3flash_traces.jsonl"


def load():
    best = {}
    for line in open(TRACES):
        r = json.loads(line)
        a = r.get("attributes") or {}
        if a.get("openinference.span.kind") != "LLM":
            continue
        idxs = [int(m.group(1)) for k in a for m in [MSG.fullmatch(k)] if m]
        if idxs:
            t = r["trace_id"]
            if t not in best or len(idxs) > best[t][0]:
                best[t] = (len(idxs), a, max(idxs))
    aw = []
    for t, (n, a, mx) in best.items():
        task = None
        snaps = []
        for i in range(mx + 1):
            role = a.get(f"llm.input_messages.{i}.message.role")
            c = a.get(f"llm.input_messages.{i}.message.content") or ""
            if role == "user":
                m = re.search(r"# Real Task Instruction\n(.*?)(?:\n\n|$)", c, re.S)
                if m:
                    task = m.group(1).strip()
            elif role == "tool" and isinstance(c, str) and len(c) >= 80:
                snaps.append(c)
        if len(snaps) >= 2 and task:
            aw.append({"task": task, "snaps": snaps})
    seen = set()
    return [x for x in aw if not (x["task"] in seen or seen.add(x["task"]))]


def measure(aw, spec):
    return measure_disjoint(
        [{"task": x["task"],
          "snaps": [lines_of(apply_spec(sn, spec) if spec else sn) for sn in x["snaps"]]}
         for x in aw])


def main():
    aw = load()
    raw = measure(aw, None)
    masked = measure(aw, ["json:release_date"])
    _bootstrap.emit(
        "Stateful API responses / AppWorld",
        {"prefix": 16.93, "pic": 0.21, "pic_proc": 0.40,
         "in_sess": 0.00, "cross_sess": 0.40},
        {"prefix_caching_pct": raw["prefix_caching_pct"], "pic_pct": raw["pic_pct"],
         "pic_proc_pct": masked["pic_pct"], "in_session": masked["in_session"],
         "cross_session": masked["cross_session"]})


if __name__ == "__main__":
    main()
