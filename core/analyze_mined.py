"""Automatic volatile-field discovery. New file.

Generalizes the hand-found backend_node_id result into a mined, per-tool
canonicalization spec: compare repeated outputs of the same tool within a
session, extract field occurrences (HTML attributes; bracketed AXTree node ids),
and score each field by the mean Jaccard overlap of its value set across
consecutive snapshots. Fields whose values almost never recur across snapshots
(low overlap, high support) are volatile -> compiled into a strip/mask spec.

Validation on two web benchmarks with different observation formats:
  * Mind2Web (HTML DOM)      -- must rediscover backend_node_id blind.
  * NNetNav-WA (WebArena AXTree) -- unseen format, no hand spec exists.
Reuse measured as position-independent 16-token content blocks (the B condition),
raw vs mined-canonical (vs hand-written canon where one exists). Both metrics.
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

from transformers import AutoTokenizer

from data import load_hf_token

TOK = {"qwen3": "Qwen/Qwen3-0.6B"}
ATTR = re.compile(r'(\w[\w-]*)="([^"]*)"')
BRACKET = re.compile(r"\[(\d+)\]")
JSONPAIR = re.compile(r'"([A-Za-z_][\w.-]*)"\s*:\s*(?:"([^"]{0,120})"|(-?\d[\d.eE+-]*))')
LOGKV = re.compile(r"\b([a-z_][\w.-]{1,30})=([^\s,;\"']{1,80})")
YAMLKV = re.compile(r"^\s*(?:- )?([A-Za-z_][\w./-]*):\s+(\S.*?)\s*$", re.M)


def pos_extractor(text, d, header_tokens=6, min_tokens=8):
    """Format adapter for headered whitespace-separated logs (syslog/BGL style):
    token position i of each line is field pos:i. Opt-in via extra=."""
    for line in text.split("\n"):
        toks = line.split()
        if len(toks) >= min_tokens:
            for i in range(header_tokens):
                d[f"pos:{i}"].add(toks[i])


def field_values(text, extra=()):
    d = defaultdict(set)
    for m in ATTR.finditer(text):
        d["attr:" + m.group(1)].add(m.group(2))
    for m in BRACKET.finditer(text):
        d["bracket_id"].add(m.group(1))
    for m in JSONPAIR.finditer(text):
        d["json:" + m.group(1)].add(m.group(2) if m.group(2) is not None else m.group(3))
    for m in LOGKV.finditer(text):
        d["kv:" + m.group(1)].add(m.group(2))
    for m in YAMLKV.finditer(text):
        d["yaml:" + m.group(1)].add(m.group(2).strip('"'))
    for fn in extra:
        fn(text, d)
    return d


# Machine-generated value shapes: integers, hex/uuid-ish ids, coordinate/dimension
# tuples ("110,607.39,264,78", "1024x768"), ISO-8601 timestamps. Every
# alternative requires a digit.
MACHINE_VALUE = re.compile(
    r"\d+"
    r"|0[xX][0-9a-fA-F]+"
    r"|[0-9a-fA-F]{8,}"
    r"|[0-9a-fA-F][0-9a-fA-F:\-]{7,}"
    r"|\d[\d.,;x\s\-]*\d"
    r"|\d{4}-\d{2}-\d{2}([T ][\d:.\-]+Z?)?"
)


def churn_stats(sessions, min_support=200, sample_cap=2000, extra=()):
    stats = defaultdict(lambda: [0.0, 0])
    occ = Counter()
    samples = defaultdict(list)
    for snaps in sessions:
        fvs = [field_values(s, extra) for s in snaps]
        for fv in fvs:
            for f, vals in fv.items():
                occ[f] += len(vals)
                if len(samples[f]) < sample_cap:
                    samples[f].extend(v for v in list(vals)[:50] if v.strip())
        for a, b in zip(fvs, fvs[1:]):
            for f in set(a) & set(b):
                j = len(a[f] & b[f]) / max(1, len(a[f] | b[f]))
                stats[f][0] += j
                stats[f][1] += 1
    out = {}
    for f, (sj, npairs) in stats.items():
        if not npairs or occ[f] < min_support:
            continue
        vals = samples[f][:sample_cap]
        machine = sum(1 for v in vals if MACHINE_VALUE.fullmatch(v.strip()))
        # identifiers/counters are mostly unique; state numerics ("2" guests) come
        # from a small domain and repeat -- low distinct ratio, never maskable
        distinct = len(set(vals)) / len(vals) if vals else 0.0
        out[f] = {"mean_jaccard": round(sj / npairs, 3), "occurrences": occ[f],
                  "machine_value_frac": round(machine / len(vals), 2) if vals else 0.0,
                  "distinct_value_ratio": round(distinct, 2)}
    return out


def line_dedup(sessions, spec):
    seen = set()
    dup = tot = 0
    for snaps in sessions:
        for s in snaps:
            if spec:
                s = apply_spec(s, spec)
            for line in s.split("\n"):
                line = line.strip()
                if not line:
                    continue
                n = len(line)
                tot += n
                k = hash(line)
                if k in seen:
                    dup += n
                else:
                    seen.add(k)
    return 100.0 * dup / tot if tot else 0.0


def mine(sessions, min_support=200, churn_ceiling=0.9, min_gain=0.5, extra=()):
    """Candidates = fields with any cross-snapshot churn; each is kept only if
    masking it measurably increases duplication (counterfactual verification on a
    cheap line-hash proxy). Self-calibrating: no stability threshold to tune."""
    report = churn_stats(sessions, min_support, extra=extra)
    candidates = sorted(
        (f for f, st in report.items()
         if st["mean_jaccard"] < churn_ceiling
         # safety gate, lesson 2: mask churn that is not state, never content.
         # (i) values must be machine-shaped (ids/counters/coords, not language)
         and st["machine_value_frac"] >= 0.9
         # (ii) values must be identifier-like (mostly unique); machine-shaped
         # state from a small domain (quantity="2") repeats and is protected
         and st["distinct_value_ratio"] >= 0.05),
        key=lambda f: report[f]["mean_jaccard"])
    spec = []
    base = line_dedup(sessions, spec)
    for f in candidates:
        g = line_dedup(sessions, spec + [f]) - base
        report[f]["dedup_gain_pts"] = round(g, 2)
        if g >= min_gain:
            spec.append(f)
            base += g
            report[f]["kept"] = True
        else:
            report[f]["kept"] = False
    return sorted(spec), dict(sorted(report.items(), key=lambda kv: kv[1]["mean_jaccard"]))


def apply_spec(text, spec):
    for f in spec:
        if f.startswith("attr:"):
            text = re.sub(r'\s*' + re.escape(f[5:]) + r'="[^"]*"', "", text)
        elif f == "bracket_id":
            text = BRACKET.sub("[N]", text)
        elif f.startswith("json:"):
            k = re.escape(f[5:])
            text = re.sub(r'"' + k + r'"\s*:\s*(?:"[^"]{0,120}"|-?\d[\d.eE+-]*)',
                          '"' + f[5:] + '":"<V>"', text)
        elif f.startswith("kv:"):
            k = re.escape(f[3:])
            text = re.sub(r"\b" + k + r"=[^\s,;\"']{1,80}", f[3:] + "=<V>", text)
        elif f.startswith("yaml:"):
            k = re.escape(f[5:])
            text = re.sub(r"(?m)^(\s*(?:- )?)" + k + r":\s+\S.*$",
                          r"\g<1>" + f[5:] + ": <V>", text)
        elif f.startswith("pos:"):
            i = int(f[4:])
            lines = text.split("\n")
            for li, line in enumerate(lines):
                toks = line.split()
                if len(toks) >= 8 and i < len(toks):
                    toks[i] = "<V>"
                    lines[li] = " ".join(toks)
            text = "\n".join(lines)
    return text


# ------------------------------------------------------------- loaders ------
def sessions_mind2web(max_tasks, tries=6):
    """The streaming iterator's underlying httpx client is closed out from under
    us on long reads ("Cannot send a request, as the client has been closed"),
    so rebuild the stream and skip what we already consumed rather than lose the
    run."""
    from datasets import load_dataset
    out, consumed = [], 0
    for attempt in range(tries):
        try:
            ds = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
            for i, ex in enumerate(ds):
                if i < consumed:
                    continue
                consumed = i + 1
                snaps = [a.get("cleaned_html") or "" for a in (ex.get("actions") or [])]
                snaps = [s for s in snaps if s]
                if len(snaps) >= 2:
                    out.append(snaps)
                if len(out) >= max_tasks:
                    return out
            return out
        except Exception as e:
            print(f"  mind2web stream attempt {attempt+1}/{tries} died after "
                  f"{consumed} examples ({len(out)} kept): "
                  f"{type(e).__name__}: {str(e)[:110]}", flush=True)
            if attempt == tries - 1:
                raise
    return out


OBS = re.compile(r"OBSERVATION:\s*\n(.*?)\nURL:", re.S)


def sessions_nnetnav(max_tasks, max_lines=4000):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("stanfordnlp/nnetnav-wa", "train.jsonl", repo_type="dataset")
    by_task = defaultdict(list)
    with open(p) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            r = json.loads(line)
            m = OBS.search(r.get("prompt") or "")
            if m:
                by_task[r.get("id")].append(m.group(1))
    out = [snaps for snaps in by_task.values() if len(snaps) >= 2]
    return out[:max_tasks]


def sessions_itbench(max_tasks):
    from huggingface_hub import HfApi, hf_hub_download
    repo = "ibm-research/ITBench-Trajectories"
    files = sorted(s.rfilename for s in HfApi().dataset_info(repo).siblings
                   if s.rfilename.endswith("session.jsonl"))
    out = []
    for rf in files:
        p = hf_hub_download(repo, rf, repo_type="dataset")
        snaps = []
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            pl = r.get("payload") or {}
            if pl.get("type") != "function_call_output":
                continue
            o = pl.get("output") or ""
            try:                       # outputs arrive JSON-wrapped: {"output": ..., "metadata": ...}
                inner = json.loads(o)
                o = json.dumps(inner, ensure_ascii=False)
            except (ValueError, TypeError):
                pass
            if len(o) >= 80:
                snaps.append(o)
        if len(snaps) >= 2:
            out.append(snaps)
        if len(out) >= max_tasks:
            break
    return out


def _snapshot_dl(repo, patterns):
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, repo_type="dataset", allow_patterns=patterns,
                                 local_files_only=True)
    except Exception:
        return snapshot_download(repo, repo_type="dataset", allow_patterns=patterns)


def sessions_itbench_ciso(max_tasks):
    import glob
    root = _snapshot_dl("ibm-research/ITBench-Lite", ["snapshots/ciso/**"])
    out = []
    for scen in sorted(glob.glob(os.path.join(root, "snapshots/ciso/*/*"))):
        snaps = [open(p).read() for p in
                 sorted(glob.glob(os.path.join(scen, "static-resources*/**/*.yaml"),
                                  recursive=True))]
        snaps = [s for s in snaps if len(s) >= 80]
        if len(snaps) >= 2:
            out.append(snaps)
        if len(out) >= max_tasks:
            break
    return out


def sessions_itbench_alerts(max_tasks):
    import glob
    root = _snapshot_dl("ArtificialAnalysis/ITBench-AA", ["sre/**"])
    out = []
    for scen in sorted(glob.glob(os.path.join(root, "sre/Scenario-*"))):
        snaps = [open(p).read() for p in
                 sorted(glob.glob(os.path.join(scen, "alerts/alerts_at_*.json")))]
        snaps = [s for s in snaps if len(s) >= 80]
        if len(snaps) >= 2:
            out.append(snaps)
        if len(out) >= max_tasks:
            break
    return out


# ---------------------------------------------------------- validation ------
def block_reuse(sessions, tok, canon=None, bs=16):
    cache = set()
    reads = hit = reused = total = 0
    frac = 0.0
    for snaps in sessions:
        for s in snaps:
            if canon:
                s = canon(s)
            ids = tok(s, add_special_tokens=False)["input_ids"]
            nb = len(ids) // bs
            r = 0
            for i in range(nb):
                k = hash(tuple(ids[i * bs:(i + 1) * bs]))
                if k in cache:
                    r += bs
                else:
                    cache.add(k)
            t = nb * bs
            if not t:
                continue
            reads += 1
            if r:
                hit += 1
            reused += r
            total += t
            frac += r / t
    p = lambda a, b: round(100.0 * a / b, 2) if b else 0.0
    return {"outputs": reads, "output_hit_pct": p(hit, reads),
            "token_reuse_agg_pct": p(reused, total),
            "token_reuse_perprompt_pct": round(100.0 * frac / reads, 2) if reads else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="qwen3")
    ap.add_argument("--max-tasks", type=int, default=40)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="report/mined_canon")
    args = ap.parse_args()
    load_hf_token()
    tok = AutoTokenizer.from_pretrained(TOK.get(args.tokenizer, args.tokenizer),
                                        token=os.environ.get("HF_TOKEN"))
    results = {}
    benches = {
        "mind2web": (sessions_mind2web, ["attr:backend_node_id"]),
        "nnetnav_webarena": (sessions_nnetnav, None),
        "itbench": (sessions_itbench, None),
        "itbench_ciso": (sessions_itbench_ciso, None),
        "itbench_alerts": (sessions_itbench_alerts, None),
    }
    if args.only:
        benches = {k: v for k, v in benches.items() if k in args.only.split(",")}
    for name, (loader, hand_spec) in benches.items():
        t0 = time.time()
        sessions = loader(args.max_tasks)
        spec, report = mine(sessions)
        entry = {"sessions": len(sessions),
                 "snapshots": sum(len(s) for s in sessions),
                 "mined_spec": spec, "field_churn": report}
        entry["raw"] = block_reuse(sessions, tok)
        entry["mined"] = block_reuse(sessions, tok, lambda s: apply_spec(s, spec))
        if hand_spec:
            entry["hand"] = block_reuse(sessions, tok, lambda s: apply_spec(s, hand_spec))
        entry["elapsed_sec"] = round(time.time() - t0, 1)
        results[name] = entry
        print(f"== {name}: {entry['sessions']} sessions / {entry['snapshots']} snapshots")
        print(f"   mined spec: {spec}")
        for f, st in list(report.items())[:8]:
            print(f"     {f:28s} jaccard={st['mean_jaccard']:.3f} occ={st['occurrences']:>7d} "
                  f"machine={st['machine_value_frac']:.2f} distinct={st['distinct_value_ratio']:.2f} "
                  f"gain={st.get('dedup_gain_pts', '-')} kept={st.get('kept', '-')}")
        for cond in ("raw", "hand", "mined"):
            if cond in entry:
                c = entry[cond]
                print(f"   {cond:5s} hit {c['output_hit_pct']:6.2f}%  reuse(agg) "
                      f"{c['token_reuse_agg_pct']:6.2f}%  (pp {c['token_reuse_perprompt_pct']:.2f}%)",
                      flush=True)
    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    json.dump(results, open(args.out + ".json", "w"), indent=2)
    print("wrote", args.out + ".json")


if __name__ == "__main__":
    main()
