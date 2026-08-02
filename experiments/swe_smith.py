"""Row: Coding agents (all input) / SWE-smith (1K).

Whole-prompt reuse under the five accounting rules (analyze_rules): every
prefilled token in the denominator, the system prompt credited to prefix
caching (rule 5), PIC = >=500-token line-aligned spans from each newly appended
message, one trajectory per task. Treatment: none.

Published: Prefix 4.21%  PIC 17.06%  PIC-proc 17.06%  in-sess 2.43  cross-sess 14.64
Data: HF SWE-bench/SWE-smith-trajectories (via data_traj). Requires HF_TOKEN.
"""
import _bootstrap
from analyze_rules import run

CORPUS = "swesmith"


def main():
    r = run(CORPUS, "Qwen/Qwen3-0.6B", max_trajs=1000, one_per_task=True,
            system_as_prefix=True)
    _bootstrap.emit(
        "Coding agents (all input) / SWE-smith (1K)",
        {"prefix": 4.21, "pic": 17.06, "pic_proc": 17.06,
         "in_sess": 2.43, "cross_sess": 14.64},
        {"prefix_caching_pct": r["system_prefix_pct"], "pic_pct": r["reuse_pct"],
         "in_session": r["in_session_pct"], "cross_session": r["cross_session_pct"]})


if __name__ == "__main__":
    main()
