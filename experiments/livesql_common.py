"""Shared schema-context measurement for the LiveSQLBench and BIRD-INTERACT rows."""
import json
import os

from transformers import AutoTokenizer

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


def db_context_units(root, db):
    d = os.path.join(root, db)
    units = []
    p = os.path.join(d, db + "_schema.txt")
    if os.path.exists(p):
        units += [b for b in open(p).read().split("\n\n") if b.strip()]
    p = os.path.join(d, db + "_column_meaning_base.json")
    if os.path.exists(p):
        cm = json.load(open(p))
        units += [f'{k}: {v}' for k, v in sorted(cm.items())]
    p = os.path.join(d, db + "_kb.jsonl")
    if os.path.exists(p):
        units += [l.strip() for l in open(p) if l.strip()]
    return units


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


def prefix_measure(prompts, schema_first):
    cache = set()
    reu = tot = 0
    for p in prompts:
        text = (p["ctx"] + "\n" + p["head"]) if schema_first else (p["head"] + "\n" + p["ctx"])
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
    reu = tot = w = x = 0
    for p in prompts:
        tot += nt(p["head"]) + sum(nt(u) for u in p["units"])
        for sp, n in spans500(p["units"]):
            k = hash(sp)
            prev = seen.get(k)
            if prev is None:
                seen[k] = p["task"]
            elif prev == p["task"]:
                reu += n
                w += n
            else:
                reu += n
                x += n
    P = lambda a: round(100.0 * a / tot, 2) if tot else 0
    return {"total": P(reu), "same_task": P(w), "cross_task": P(x)}
