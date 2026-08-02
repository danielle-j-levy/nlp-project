"""Row: Database schemas / LiveSQLBench.

One task = one session; snapshot = the selected database's context (schema DDL +
column meanings + KB). Prefix caching reported schema-first / schema-after; PIC
packs the context into >=500-token spans reused across tasks on the same DB.
Treatment: none.

Published: Prefix 95.9/0.0%  PIC 95.86%  PIC-proc 95.86%  in-sess 0.00  cross-sess 95.86
Data: local data/livesqlbench/large-v1/ (livesqlbench_large_v1_data.jsonl + per-db context).
"""
import json
import os

import _bootstrap
from livesql_common import db_context_units, prefix_measure, pic_measure

ROOT = "data/livesqlbench/large-v1"


def main():
    tasks = [json.loads(l) for l in open(os.path.join(ROOT, "livesqlbench_large_v1_data.jsonl"))]
    ctx_cache = {}
    prompts = []
    for t in tasks:
        db = t["selected_database"]
        if db not in ctx_cache:
            u = db_context_units(ROOT, db)
            ctx_cache[db] = (u, "\n".join(u))
        u, ctx = ctx_cache[db]
        prompts.append({"task": t["instance_id"], "db": db, "units": u, "ctx": ctx,
                        "head": "Task " + t["instance_id"] + "\nQuestion: " + t["query"]})
    print(f"livesqlbench: {len(prompts)} tasks, {len(ctx_cache)} dbs", flush=True)
    pic = pic_measure(prompts)
    _bootstrap.emit(
        "Database schemas / LiveSQLBench",
        {"prefix": "95.9/0.0", "pic": 95.86, "pic_proc": 95.86,
         "in_sess": 0.00, "cross_sess": 95.86},
        {"prefix_schema_first": prefix_measure(prompts, True),
         "prefix_schema_after": prefix_measure(prompts, False),
         "pic_pct": pic["total"], "in_session": pic["same_task"],
         "cross_session": pic["cross_task"]})


if __name__ == "__main__":
    main()
