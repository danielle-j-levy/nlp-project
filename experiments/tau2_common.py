"""Shared tau2-bench loader for the three domain rows (airline/retail/telecom)."""
import glob
import json
import os
import re

from pic_rules import lines_of
from prefix_disjoint import measure_disjoint

STD = re.compile(r"_(airline|retail|telecom)_")


def run_domain(domain):
    fs = [f for f in sorted(glob.glob("data/tau2/final/*.json"))
          if (m := STD.search(os.path.basename(f))) and m.group(1) == domain]
    sessions = []
    for f in fs:
        for s in (json.load(open(f)).get("simulations") or []):
            snaps = [c for msg in (s.get("messages") or [])
                     if msg.get("role") == "tool" and not msg.get("error")
                     and isinstance(c := msg.get("content"), str) and len(c) >= 80]
            if len(snaps) >= 2:
                sessions.append({"task": s.get("task_id"), "snaps": snaps})
    sessions.sort(key=lambda s: str(s["task"]))
    seen = set()
    dd = [s for s in sessions if not (s["task"] in seen or seen.add(s["task"]))]
    print(f"tau2 {domain}: {len(fs)} files, {len(sessions)} sims -> {len(dd)} tasks",
          flush=True)
    return measure_disjoint(
        [{"task": s["task"], "snaps": [lines_of(x) for x in s["snaps"]]} for s in dd])
