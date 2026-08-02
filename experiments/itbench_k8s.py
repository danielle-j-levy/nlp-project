"""Row: Operational telemetry / ITBench infra compliance (K8s).

Kubernetes resource snapshots (CISO compliance). Untreated PIC already the best
in the family, so no treatment is applied. Treatment: none.

Published: Prefix 0.71%  PIC 26.73%  PIC-proc 26.73%  in-sess 26.73  cross-sess 0.00
Data: HF ibm-research/ITBench-Lite (snapshots/ciso). Requires HF_TOKEN.
"""
import _bootstrap
from pic_rules import lines_of
from prefix_disjoint import measure_disjoint
from analyze_mined import sessions_itbench_ciso


def main():
    s = sessions_itbench_ciso(200)
    u = [{"task": i, "snaps": [lines_of(sn) for sn in snaps]}
         for i, snaps in enumerate(s)]
    d = measure_disjoint(u)
    _bootstrap.emit(
        "Operational telemetry / ITBench infra compliance (K8s)",
        {"prefix": 0.71, "pic": 26.73, "pic_proc": 26.73,
         "in_sess": 26.73, "cross_sess": 0.00}, d)


if __name__ == "__main__":
    main()
