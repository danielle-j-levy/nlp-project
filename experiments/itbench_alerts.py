"""Row: Operational telemetry / ITBench alerts.

Alert payloads. Untreated PIC; no treatment applied. Treatment: none.

Published: Prefix 0.08%  PIC 16.93%  PIC-proc 16.93%  in-sess 16.93  cross-sess 0.00
Data: HF ITBench alerts (Scenario-*/alerts/*.json). Requires HF_TOKEN.
"""
import _bootstrap
from pic_rules import lines_of
from prefix_disjoint import measure_disjoint
from analyze_mined import sessions_itbench_alerts


def main():
    s = sessions_itbench_alerts(40)
    u = [{"task": i, "snaps": [lines_of(sn) for sn in snaps]}
         for i, snaps in enumerate(s)]
    d = measure_disjoint(u)
    _bootstrap.emit(
        "Operational telemetry / ITBench alerts",
        {"prefix": 0.08, "pic": 16.93, "pic_proc": 16.93,
         "in_sess": 16.93, "cross_sess": 0.00}, d)


if __name__ == "__main__":
    main()
