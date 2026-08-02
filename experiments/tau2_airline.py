"""Row: Stateful API responses / tau2 airline.

tau2-bench customer-support simulations; tool outputs are per-domain API
responses. One simulation = one session, pooled across agent models, deduplicated
to one trajectory per task. Treatment: none.

Published: Prefix 12.81%  PIC 1.31%  PIC-proc 1.31%  in-sess 1.31  cross-sess 0.00
Data: local data/tau2/final/*airline*.json (sierra-research/tau2-bench).
"""
import _bootstrap
from tau2_common import run_domain


def main():
    d = run_domain("airline")
    _bootstrap.emit("Stateful API responses / tau2 airline",
                    {"prefix": 12.81, "pic": 1.31, "pic_proc": 1.31,
                     "in_sess": 1.31, "cross_sess": 0.00}, d)


if __name__ == "__main__":
    main()
