"""Web-agent rows on the coding rows' denominator convention.

web_rules5_combined counts each step's whole prompt, so the fixed instruction
head is counted once per step. The coding rows count every token once per
session (rule 5: a turn contributes only its newly appended text). Here the head
is counted once per session and each observation once, and prefix credit is
adjusted to match: within a session the head is no longer in the denominator
after step 0, so only the part of the served run that reaches past the head --
the genuinely identical leading portion of the observation -- is credited.
A session's first request may still hit the head from an earlier session, which
is cross-request reuse of an identical prefix and is kept.
"""
import json, re
from collections import Counter, defaultdict
from data import load_hf_token; load_hf_token()
import pic_rules
from pic_rules import nt, spans_merge

BS = 16
OBS = re.compile(r"OBSERVATION:\s*\n", re.S)
BR = re.compile(r"\[(\d+)\]")


def h64(s):
    h = 1469598103934665603
    for c in s.encode():
        h = ((h ^ c) * 1099511628211) & (1 << 64) - 1
    return h


def replace_handles(snapshot):
    occ = Counter(); out = []
    for line in snapshot.split("\n"):
        cleaned = BR.sub("[.]", line); occ[cleaned] += 1
        o = occ[cleaned] - 1; i = [0]
        def sub(m):
            aid = h64(f"{cleaned}|{o}|{i[0]}"); i[0] += 1
            return f"[a{aid:016x}]"
        out.append(BR.sub(sub, line))
    return "\n".join(out)


def load(repo, max_tasks=40, max_lines=4000):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "train.jsonl", repo_type="dataset")
    by = defaultdict(list)
    for i, line in enumerate(open(p)):
        if i >= max_lines: break
        r = json.loads(line)
        if r.get("prompt"): by[r.get("id")].append(r["prompt"])
    return [v for v in by.values() if len(v) >= 2][:max_tasks]


def run(repo, treat):
    nt("warm"); tok = pic_rules._tok
    prefix_cache = set(); seen = {}
    denom = prefix_served = pic_in = pic_cross = head_once = 0
    for sid, prompts in enumerate(load(repo)):
        for step, prompt in enumerate(prompts):
            m = OBS.search(prompt)
            head = prompt[:m.end()] if m else prompt
            obs = prompt[m.end():] if m else ""
            if treat and obs: obs = treat(obs)
            text = head + obs
            ids = tok(text, add_special_tokens=False)["input_ids"]
            head_n = len(tok(head, add_special_tokens=False)["input_ids"])
            obs_n = len(ids) - head_n
            h = 0; chain = []; hits = 0; counting = True
            for i in range(len(ids) // BS):
                h = hash((h, tuple(ids[i*BS:(i+1)*BS])))
                chain.append(h)
                if counting:
                    if h in prefix_cache: hits += 1
                    else: counting = False
            for h2 in chain: prefix_cache.add(h2)
            served = hits * BS
            # rule 5, coding-row convention: the head enters the denominator once
            # per session, so it may only be credited on that one occurrence
            if step == 0:
                denom += head_n + obs_n
                head_once += head_n
                prefix_served += min(served, head_n) + max(0, served - head_n)
            else:
                denom += obs_n
                prefix_served += max(0, served - head_n)
            skip = max(0, served - head_n)
            kept, acc = [], 0
            for l in [x for x in obs.split("\n") if x.strip()]:
                t = nt(l)
                if acc + t <= skip: acc += t; continue
                kept.append(l)
            for sp, tn in spans_merge(kept):
                k = hash(sp); p = seen.get(k)
                if p is None: seen[k] = sid
                elif p == sid: pic_in += tn
                else: pic_cross += tn
    P = lambda a: round(100.0 * a / denom, 2) if denom else 0
    return {"stream_tokens": denom, "head_counted_once_tokens": head_once,
            "prefix_caching_pct": P(prefix_served),
            "pic_in_sess_pct": P(pic_in), "pic_cross_sess_pct": P(pic_cross),
            "pic_pct": P(pic_in + pic_cross)}
