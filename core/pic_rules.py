"""Shared PIC measurement under the paper's four accounting rules:

1. Reuse counts from the second occurrence of a span onward; the first
   occurrence populates the cache and contributes zero.
2. The denominator is EVERY token of the evaluated trajectories -- including
   outputs too small to ever form a span and relocated sidecar tokens --
   so the percentage is a realistic lower bound over the whole stream.
3. Trajectories are de-duplicated to one per distinct task; the all-runs
   figure may be reported alongside, labeled as task-repetition reuse.
4. Spans are >=500 tokens: units pack greedily until the count crosses 500;
   a tail that cannot reach 500 merges into the previous span; whole outputs
   under 500 tokens never form a span (but still count in the denominator).
"""

from transformers import AutoTokenizer

MIN = 500
_tok = None
_tc = {}


def nt(s):
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    v = _tc.get(s)
    if v is None:
        v = len(_tok(s, add_special_tokens=False)["input_ids"])
        _tc[s] = v
    return v


def spans_merge(units):
    out = []
    buf = []
    n = 0
    for u in units:
        buf.append(u)
        n += nt(u)
        if n >= MIN:
            out.append(["".join(buf), n])
            buf = []
            n = 0
    if buf and out:
        out[-1][0] += "".join(buf)
        out[-1][1] += n
    return out


def lines_of(t):
    """Units are lines WITH their trailing newline, and blank lines are kept.

    A newline is a real token the model prefills. Dropping it understates the
    denominator (rule 2) and, because spans then have to pack more lines before
    crossing the 500-token floor, changes which spans exist at all -- so it
    perturbs the numerator too. Byte-faithful units keep both sides of the ratio
    on the text the model actually sees."""
    return [l + "\n" for l in t.split("\n")]


def dedup_tasks(sessions):
    seen = set()
    return [s for s in sessions if not (s["task"] in seen or seen.add(s["task"]))]


def measure(sessions, sidecar=0):
    """sessions: [{"task": id, "snaps": [[unit, ...], ...]}, ...]"""
    seen = {}
    full = reu = w = xt = xo = span_tok = 0
    for sid, sess in enumerate(sessions):
        for units in sess["snaps"]:
            full += sum(nt(u) for u in units)
            for sp, n in spans_merge(units):
                span_tok += n
                k = hash(sp)
                p = seen.get(k)
                if p is None:
                    seen[k] = sid
                elif p == sid:
                    reu += n
                    w += n
                elif sessions[p]["task"] == sess["task"]:
                    reu += n
                    xt += n
                else:
                    reu += n
                    xo += n
    denom = full + sidecar
    P = lambda a: round(100.0 * a / denom, 2) if denom else 0
    return {"total": P(reu), "same_session": P(w),
            "cross_run_same_task": P(xt), "cross_task": P(xo),
            "stream_tokens": denom, "reused_tokens": reu,
            "span_coverage_pct": round(100.0 * span_tok / denom, 1) if denom else 0,
            "sessions": len(sessions)}


def report(name, sessions, sidecar=0, dedup_sidecar=None):
    allr = measure(sessions, sidecar)
    dd = dedup_tasks(sessions)
    ddr = measure(dd, sidecar if dedup_sidecar is None else dedup_sidecar)
    return {"all_runs": allr, "one_run_per_task": ddr,
            "duplicate_runs": len(sessions) - len(dd)}
