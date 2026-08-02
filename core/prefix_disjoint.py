"""PIC credit strictly beyond what a prefix cache serves (rule 5, cross-session).

Rule 5 excludes reuse ordinary prefix caching already provides. Skipping only the
within-session prefix is not enough: when two sessions open on the same text a
shared prefix cache serves that leading run at position 0, so it is not PIC's to
claim -- the system prompt being the archetypal case.

Two invariants matter and are easy to get wrong:

* Spans are packed PER SNAPSHOT. Each snapshot is a separate tool observation
  arriving at its own turn with assistant text in between, so a span must never
  straddle two of them; packing over a flattened session invents spans that
  never appear contiguously in any prompt and collapses real reuse.
* Span boundaries are decided before the prefix cut is applied, and the cut only
  decides which spans may score. Re-packing from the cut would give each session
  its own span grid and destroy the alignment that treatments like bucket
  packing exist to create.
"""
from pic_rules import MIN, nt


def _h(s):
    h = 1469598103934665603
    for c in s.encode():
        h = ((h ^ c) * 1099511628211) & ((1 << 64) - 1)
    return h


def _spans(units, base):
    """[(global_start, global_end, text, tokens)] over one snapshot; tail merged."""
    out, buf, n, start = [], [], 0, 0
    for i, u in enumerate(units):
        buf.append(u); n += nt(u)
        if n >= MIN:
            out.append([base + start, base + i + 1, "".join(buf), n])
            buf = []; n = 0; start = i + 1
    if buf and out:
        out[-1][1] = base + len(units); out[-1][2] += "".join(buf); out[-1][3] += n
    return out


def measure_disjoint(sessions, sidecar=0):
    """sessions: [{"task": id, "snaps": [[unit, ...], ...]}, ...]"""
    prefix_seen = set()
    span_seen = {}
    denom = prefix_tok = reu = w = x = 0
    for sid, sess in enumerate(sessions):
        snaps = sess["snaps"]
        flat = [u for sn in snaps for u in sn]
        denom += sum(nt(u) for u in flat)
        # leading run this session shares with some earlier session's opening
        cut, h = 0, 0
        for i, u in enumerate(flat):
            h = ((h * 1099511628211) ^ _h(u)) & ((1 << 64) - 1)
            if h in prefix_seen:
                cut = i + 1
            else:
                break
        prefix_tok += sum(nt(u) for u in flat[:cut])
        h = 0
        for u in flat[:2000]:
            h = ((h * 1099511628211) ^ _h(u)) & ((1 << 64) - 1)
            prefix_seen.add(h)
        base = 0
        for sn in snaps:
            for a, b, sp, n in _spans(sn, base):
                if b <= cut:                 # wholly inside the prefix-served run
                    continue
                k = _h(sp)
                p = span_seen.get(k)
                if p is None:
                    span_seen[k] = sid
                elif p == sid:
                    reu += n; w += n
                else:
                    reu += n; x += n
            base += len(sn)
    denom += sidecar
    P = lambda a: round(100.0 * a / denom, 2) if denom else 0
    return {"prefix_caching_pct": P(prefix_tok), "pic_pct": P(reu),
            "in_session": P(w), "cross_session": P(x),
            "stream_tokens": denom, "prefix_tokens": prefix_tok,
            "pic_tokens": reu, "sidecar": sidecar, "sessions": len(sessions)}
