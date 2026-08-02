"""Row: Stateful API responses / tau2 telecom.

Same pipeline as the airline row (tau2_common.run_domain). Treatment: none.

Published: Prefix 60.70%  PIC 3.73%  PIC-proc 3.73%  in-sess 0.00  cross-sess 3.73
Data: local data/tau2/final/*telecom*.json (sierra-research/tau2-bench).
"""
import _bootstrap
from tau2_common import run_domain

if __name__ == "__main__":
    d = run_domain("telecom")
    _bootstrap.emit("Stateful API responses / tau2 telecom",
                    {"prefix": 60.70, "pic": 3.73, "pic_proc": 3.73,
                     "in_sess": 0.00, "cross_sess": 3.73}, d)
