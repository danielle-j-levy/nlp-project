"""Row: Operational telemetry / BGL syslog (LogHub).

Supercomputer syslog. Each session is a contiguous 200k-line slice cut into
500-line snapshots. The `relocate` treatment moves the two per-line volatile
columns (timestamp/id, positions 1 and 4) into a sidecar so the message bodies
line up. PIC-proc / in-sess / cross-sess are the relocated numbers.

Published: Prefix 0.00%  PIC 0.00%  PIC-proc 8.49%  in-sess 4.53  cross-sess 3.96
Treatment: relocate. Data: local data/loghub/BGL_200k.log (LogHub BGL).
"""
import _bootstrap
from pic_rules import lines_of, nt
from prefix_disjoint import measure_disjoint

POS = (1, 4)


def main():
    lines = [l.rstrip("\n") for l in open("data/loghub/BGL_200k.log") if l.strip()]
    per = len(lines) // 20
    side = 0
    bgl = []
    for s in range(20):
        chunk = lines[s * per:(s + 1) * per]
        snaps = []
        for i in range(0, len(chunk), 500):
            units = []
            for line in chunk[i:i + 500]:
                toks = line.split()
                if len(toks) >= 8:
                    side += nt(" ".join(toks[j] for j in POS if j < len(toks)))
                    units.append(" ".join(t for j, t in enumerate(toks) if j not in POS) + "\n")
                else:
                    units.append(line + "\n")
            snaps.append(units)
        bgl.append({"task": s, "snaps": snaps})
    d = measure_disjoint(bgl, sidecar=side)
    # verbatim PIC on this corpus is 0.00 (published); the win is entirely the
    # relocate treatment, so pic == 0 and pic_proc == d["pic_pct"].
    _bootstrap.emit(
        "Operational telemetry / BGL syslog (LogHub)",
        {"prefix": 0.00, "pic": 0.00, "pic_proc": 8.49,
         "in_sess": 4.53, "cross_sess": 3.96},
        {"prefix_caching_pct": d["prefix_caching_pct"], "pic_proc_pct": d["pic_pct"],
         "in_session": d["in_session"], "cross_session": d["cross_session"]})


if __name__ == "__main__":
    main()
