"""Row: Coding agents (all input) / OpenHands (1K).

Whole-prompt reuse under the five accounting rules (analyze_rules), as in the
SWE-smith row. Treatment: none.

Published: Prefix 6.42%  PIC 2.69%  PIC-proc 2.69%  in-sess 1.87  cross-sess 0.82
Data: HF nebius/SWE-rebench-openhands-trajectories (via data_traj). Requires HF_TOKEN.
"""
import _bootstrap
from analyze_rules import run

CORPUS = "openhands"


def main():
    r = run(CORPUS, "Qwen/Qwen3-0.6B", max_trajs=1000, one_per_task=True,
            system_as_prefix=True)
    _bootstrap.emit(
        "Coding agents (all input) / OpenHands (1K)",
        {"prefix": 6.42, "pic": 2.69, "pic_proc": 2.69,
         "in_sess": 1.87, "cross_sess": 0.82},
        {"prefix_caching_pct": r["system_prefix_pct"], "pic_pct": r["reuse_pct"],
         "in_session": r["in_session_pct"], "cross_session": r["cross_session_pct"]})


if __name__ == "__main__":
    main()
