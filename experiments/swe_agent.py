"""Row: Coding agents (all input) / SWE-agent (1K).

Whole-prompt reuse under the five accounting rules (analyze_rules). Treatment: none.
Note: this is the nebius/SWE-agent-trajectories corpus, not SWE-bench Verified.

Published: Prefix 7.96%  PIC 19.27%  PIC-proc 19.27%  in-sess 18.99  cross-sess 0.28
Data: HF SWE-agent trajectories (via data_traj corpus "sweagent"). Requires HF_TOKEN.
"""
import _bootstrap
from analyze_rules import run

CORPUS = "sweagent"


def main():
    r = run(CORPUS, "Qwen/Qwen3-0.6B", max_trajs=1000, one_per_task=True,
            system_as_prefix=True)
    _bootstrap.emit(
        "Coding agents (all input) / SWE-agent (1K)",
        {"prefix": 7.96, "pic": 19.27, "pic_proc": 19.27,
         "in_sess": 18.99, "cross_sess": 0.28},
        {"prefix_caching_pct": r["system_prefix_pct"], "pic_pct": r["reuse_pct"],
         "in_session": r["in_session_pct"], "cross_session": r["cross_session_pct"]})


if __name__ == "__main__":
    main()
