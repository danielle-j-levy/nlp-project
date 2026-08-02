"""Row: Chat prompts / WildChat-1M (40K).

Each prompt is one session carrying one snapshot. Prefix caching is the leading
run shared with an earlier prompt; PIC spans (>=500 byte-faithful line units,
reuse from the second occurrence) are cut from the remainder. Treatment: none.

Published: Prefix 26.62%  PIC 3.28%  PIC-proc 3.28%  in-sess 0.02  cross-sess 3.27
Data: HF allenai/WildChat-1M (streamed). Requires HF_TOKEN.
"""
import _bootstrap
from data import iter_first_turns
from pic_rules import lines_of
from prefix_disjoint import measure_disjoint

N = 40_000


def main():
    texts = [r["text"] for r in iter_first_turns(max_rows=N)]
    print(f"wildchat: {len(texts)} prompts", flush=True)
    d = measure_disjoint([{"task": i, "snaps": [lines_of(t)]}
                          for i, t in enumerate(texts)])
    _bootstrap.emit("Chat prompts / WildChat-1M (40K)",
                    {"prefix": 26.62, "pic": 3.28, "pic_proc": 3.28,
                     "in_sess": 0.02, "cross_sess": 3.27}, d)


if __name__ == "__main__":
    main()
