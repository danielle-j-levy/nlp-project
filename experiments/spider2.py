"""Row: Database schemas / Spider 2.0.

One SQL question = one session; its snapshot is the serialized DDL of the target
database. Prefix caching is reported schema-first / schema-after-context (the
"86.5/0.0" pair); PIC packs the DDL into >=500-token spans reused across
questions on the same database. Treatment: none.

Published: Prefix 86.5/0.0%  PIC 86.95%  PIC-proc 86.95%  in-sess 0.00  cross-sess 86.95
Data: local data/spider2/ (spider2-lite.jsonl, databases/**/DDL.csv, documents/).
"""
import csv
import glob
import os
import re

import _bootstrap
from transformers import AutoTokenizer

ROOT = "data/spider2"
MIN = 500
BS = 16
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tc = {}


def nt(s):
    v = tc.get(s)
    if v is None:
        v = len(tok(s, add_special_tokens=False)["input_ids"])
        tc[s] = v
    return v


ALIAS = {"sqlite-sakila": "SQLITE_SAKILA", "Db-IMDB": "DB_IMDB"}
SHARD = re.compile(r"\d{8}$")
csv.field_size_limit(10 ** 8)


def schema_units(db):
    folder = ALIAS.get(db, db)
    hits = glob.glob(os.path.join(ROOT, "databases", "*", folder))
    if not hits:
        return None
    units = []
    seen_shard = set()
    for ddl in sorted(glob.glob(os.path.join(hits[0], "**", "DDL.csv"), recursive=True)):
        rows = list(csv.DictReader(open(ddl)))
        for r in sorted(rows, key=lambda r: r["table_name"]):
            name = r["table_name"]
            base = SHARD.sub("<DATE>", name)
            if base != name:
                if base in seen_shard:
                    continue
                seen_shard.add(base)
            units.append(r.get("DDL") or r.get("ddl") or "")
    return [u for u in units if u] or None


def load_prompts():
    import json
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "spider2-lite.jsonl"))]
    prompts = []
    for q in qs:
        u = schema_units(q["db"])
        if not u:
            continue
        doc = ""
        if q.get("external_knowledge"):
            p = os.path.join(ROOT, "documents", q["external_knowledge"])
            if os.path.exists(p):
                doc = open(p).read()
        prompts.append({"db": q["db"], "iid": q["instance_id"], "units": u,
                        "question": q["question"], "doc": doc})
    return prompts


def spans500(units):
    out, buf, n = [], [], 0
    for u in units:
        buf.append(u)
        n += nt(u)
        if n >= MIN:
            out.append(["\n".join(buf), n])
            buf = []
            n = 0
    if buf and out:
        out[-1][0] += "\n" + "\n".join(buf)
        out[-1][1] += n
    return out


def stream_ids(p, schema_first):
    head = "You are a SQL agent.\nTask " + p["iid"] + "\nQuestion: " + p["question"] + "\n" + p["doc"]
    schema = "\n".join(p["units"])
    text = (schema + "\n" + head) if schema_first else (head + "\n" + schema)
    return tok(text, add_special_tokens=False)["input_ids"]


def prefix_measure(prompts, schema_first):
    cache = set()
    reu = tot = 0
    for p in prompts:
        ids = stream_ids(p, schema_first)
        h = 0
        for i in range(len(ids) // BS):
            h = hash((h, tuple(ids[i * BS:(i + 1) * BS])))
            tot += BS
            if h in cache:
                reu += BS
            else:
                cache.add(h)
    return round(100.0 * reu / tot, 2) if tot else 0.0


def pic_measure(prompts):
    seen = {}
    reu = tot = w = x = 0
    for sid, p in enumerate(prompts):
        head = "You are a SQL agent.\nTask " + p["iid"] + "\nQuestion: " + p["question"] + "\n" + p["doc"]
        tot += nt(head) + sum(nt(u) for u in p["units"])
        for sp, n in spans500(p["units"]):
            k = hash(sp)
            prev = seen.get(k)
            if prev is None:
                seen[k] = sid
            elif prev == sid:
                reu += n
                w += n
            else:
                reu += n
                x += n
    P = lambda a: round(100.0 * a / tot, 2) if tot else 0
    return {"total": P(reu), "same_session": P(w), "cross_sessions": P(x)}


def main():
    prompts = load_prompts()
    print(f"spider2: {len(prompts)} questions, {len({p['db'] for p in prompts})} dbs",
          flush=True)
    pic = pic_measure(prompts)
    _bootstrap.emit(
        "Database schemas / Spider 2.0",
        {"prefix": "86.5/0.0", "pic": 86.95, "pic_proc": 86.95,
         "in_sess": 0.00, "cross_sess": 86.95},
        {"prefix_schema_first": prefix_measure(prompts, True),
         "prefix_schema_after": prefix_measure(prompts, False),
         "pic_pct": pic["total"], "in_session": pic["same_session"],
         "cross_session": pic["cross_sessions"]})


if __name__ == "__main__":
    main()
