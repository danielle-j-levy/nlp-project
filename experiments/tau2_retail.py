"""Row: Stateful API responses / tau2 retail.

Same pipeline as the airline row (tau2_common.run_domain). Treatment: none.

Published: Prefix 23.28%  PIC 23.08%  PIC-proc 23.08%  in-sess 0.33  cross-sess 22.75
Data: local data/tau2/final/*retail*.json (sierra-research/tau2-bench).
"""
import _bootstrap
from tau2_common import run_domain

if __name__ == "__main__":
    d = run_domain("retail")
    _bootstrap.emit("Stateful API responses / tau2 retail",
                    {"prefix": 23.28, "pic": 23.08, "pic_proc": 23.08,
                     "in_sess": 0.33, "cross_sess": 22.75}, d)
