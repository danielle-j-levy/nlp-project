"""Row: Conversational SQL / BIRD-INTERACT.

Multi-turn SQL over a fixed per-database context. Each turn (ambiguous query +
optional follow-up) is a session sharing that database's schema context. Prefix
caching reported schema-first / schema-after; PIC packs the context into
>=500-token spans reused across turns/tasks on the same DB. Treatment: none.

Published: Prefix 97.9/0.0%  PIC 97.87%  PIC-proc 97.87%  in-sess 1.84  cross-sess 96.03
Data: local data/birdinteract/{lite,full}/**/bird_interact_data.jsonl
"""
import glob
import json
import os

import _bootstrap
from livesql_common import db_context_units, prefix_measure, pic_measure


def load(root):
    f = glob.glob(os.path.join(root, "**", "bird_interact_data.jsonl"), recursive=True)
    if not f:
        return None
    tasks = [json.loads(l) for l in open(f[0])]
    base = os.path.dirname(f[0])
    ctx_cache = {}
    prompts = []
    for t in tasks:
        db = t["selected_database"]
        if db not in ctx_cache:
            u = db_context_units(base, db)
            ctx_cache[db] = (u, "\n".join(u))
        u, ctx = ctx_cache[db]
        turns = [("t1", t.get("amb_user_query") or t["query"])]
        fu = (t.get("follow_up") or {}).get("query")
        if fu:
            turns.append(("t2", fu))
        for tag, q in turns:
            prompts.append({"task": t["instance_id"], "db": db, "units": u, "ctx": ctx,
                            "head": f"Task {t['instance_id']} {tag}\nQuestion: {q}"})
    return prompts


def main():
    computed = {}
    for name, root in (("lite", "data/birdinteract/lite"),
                       ("full", "data/birdinteract/full")):
        prompts = load(root)
        if prompts is None:
            print(f"birdinteract_{name}: data file not found", flush=True)
            continue
        pic = pic_measure(prompts)
        computed[name] = {
            "prefix_schema_first": prefix_measure(prompts, True),
            "prefix_schema_after": prefix_measure(prompts, False),
            "pic_pct": pic["total"], "in_session": pic["same_task"],
            "cross_session": pic["cross_task"]}
        print(f"birdinteract_{name}: {len(prompts)} turns, "
              f"{len({p['task'] for p in prompts})} tasks", flush=True)
    _bootstrap.emit(
        "Conversational SQL / BIRD-INTERACT (lite is the table row)",
        {"prefix": "97.9/0.0", "pic": 97.87, "pic_proc": 97.87,
         "in_sess": 1.84, "cross_sess": 96.03},
        computed.get("lite", computed))


if __name__ == "__main__":
    main()
