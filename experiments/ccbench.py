"""Row: Coding agents (all input) / CC-Bench (74).

Whole-prompt reuse under the five accounting rules (analyze_rules). Claude Code
carries no separate system prompt, so prefix caching is 0. Treatment: none.

Published: Prefix 0.00%  PIC 16.43%  PIC-proc 16.43%  in-sess 7.79  cross-sess 8.64
Data: HF zai-org/CC-Bench-trajectories (via data_traj). Requires HF_TOKEN.
"""
import _bootstrap
from analyze_rules import run

CORPUS = "ccbench"


def main():
    r = run(CORPUS, "Qwen/Qwen3-0.6B", max_trajs=1000, one_per_task=True,
            system_as_prefix=True)
    _bootstrap.emit(
        "Coding agents (all input) / CC-Bench (74)",
        {"prefix": 0.00, "pic": 16.43, "pic_proc": 16.43,
         "in_sess": 7.79, "cross_sess": 8.64},
        {"prefix_caching_pct": r["system_prefix_pct"], "pic_pct": r["reuse_pct"],
         "in_session": r["in_session_pct"], "cross_session": r["cross_session_pct"]})


if __name__ == "__main__":
    main()
