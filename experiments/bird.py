"""Row: Database schemas / BIRD.

One SQL question = one session; snapshot = the target DB's CREATE statements
(verbatim DDL, sorted by object name). Prefix caching reported schema-first /
schema-after-context; PIC packs the DDL into >=500-token spans reused across
questions on the same database. Treatment: none.

Published: Prefix 93.7/0.0%  PIC 78.54%  PIC-proc 78.54%  in-sess 0.00  cross-sess 78.54
Data: HF birdsql/bird_sql_dev_20251106 + schemas cached at data/bird/schemas.json.
Requires HF_TOKEN (first run builds the schema cache from bird-corpus-validation).
"""
import json
import os
import sqlite3

import _bootstrap
from pic_rules import nt, spans_merge, MIN
import pic_rules

BS = 16
DBS = ["california_schools", "card_games", "codebase_community",
       "debit_card_specializing", "european_football_2", "financial",
       "formula_1", "student_club", "superhero", "thrombosis_prediction",
       "toxicology"]


def build_schemas(path="data/bird/schemas.json"):
    if os.path.exists(path):
        return json.load(open(path))
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(path), exist_ok=True)
    schemas = {}
    for db in DBS:
        p = hf_hub_download("target-benchmark/bird-corpus-validation",
                            f"validation_database/{db}/{db}.sqlite",
                            repo_type="dataset")
        con = sqlite3.connect(p)
        schemas[db] = [r[0] for r in con.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND type IN ('table','view') ORDER BY name")]
        con.close()
    json.dump(schemas, open(path, "w"), indent=1)
    return schemas


def load_prompts():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("birdsql/bird_sql_dev_20251106",
                        "data/dev_20251106-00000-of-00001.json",
                        repo_type="dataset")
    qs = json.load(open(p))
    schemas = build_schemas()
    prompts = []
    for q in qs:
        u = schemas.get(q["db_id"])
        if u:
            head = ("You are a SQL analyst.\nQuestion " + str(q["question_id"])
                    + ": " + q["question"] + "\nEvidence: " + (q.get("evidence") or ""))
            prompts.append({"db": q["db_id"], "units": u, "head": head})
    return prompts


def prefix_measure(prompts, schema_first):
    nt("warm")
    tok = pic_rules._tok
    cache = set()
    reu = tot = 0
    for p in prompts:
        schema = "\n".join(p["units"])
        text = (schema + "\n" + p["head"]) if schema_first else (p["head"] + "\n" + schema)
        ids = tok(text, add_special_tokens=False)["input_ids"]
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
    full = reu = w = x = 0
    for sid, p in enumerate(prompts):
        full += nt(p["head"]) + sum(nt(u) for u in p["units"])
        for sp, n in spans_merge(p["units"]):
            k = hash(sp)
            q = seen.get(k)
            if q is None:
                seen[k] = sid
            elif q == sid:
                reu += n
                w += n
            else:
                reu += n
                x += n
    P = lambda a: round(100.0 * a / full, 2) if full else 0
    return {"total": P(reu), "same_session": P(w), "cross_task": P(x)}


def main():
    prompts = load_prompts()
    print(f"bird: {len(prompts)} questions, {len({p['db'] for p in prompts})} dbs",
          flush=True)
    pic = pic_measure(prompts)
    _bootstrap.emit(
        "Database schemas / BIRD",
        {"prefix": "93.7/0.0", "pic": 78.54, "pic_proc": 78.54,
         "in_sess": 0.00, "cross_sess": 78.54},
        {"prefix_schema_first": prefix_measure(prompts, True),
         "prefix_schema_after": prefix_measure(prompts, False),
         "pic_pct": pic["total"], "in_session": pic["same_session"],
         "cross_session": pic["cross_task"]})


if __name__ == "__main__":
    main()
