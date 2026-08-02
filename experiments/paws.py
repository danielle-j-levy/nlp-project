"""Row: Paraphrase traffic / PAWS (40K).

Same accounting as the WildChat row: one prompt = one session = one snapshot,
prefix caching first, PIC (>=500-token line spans) on the remainder. PAWS is
near-duplicate sentence pairs, all far below the 500-token span floor, so PIC is
zero and the whole win is ordinary prefix caching. Treatment: none.

Published: Prefix 27.67%  PIC 0.00%  PIC-proc 0.00%  in-sess 0.00  cross-sess 0.00
Data: HF google-research-datasets/paws (labeled_final, streamed).
"""
import _bootstrap
from pic_rules import lines_of
from prefix_disjoint import measure_disjoint

N = 40_000


def ex_paws(cap):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/paws", "labeled_final",
                      split="train", streaming=True)
    n = 0
    for r in ds:
        for k in ("sentence1", "sentence2"):
            t = r.get(k)
            if t:
                yield t
                n += 1
                if n >= cap:
                    return


def main():
    texts = list(ex_paws(N))
    print(f"paws: {len(texts)} prompts", flush=True)
    d = measure_disjoint([{"task": i, "snaps": [lines_of(t)]}
                          for i, t in enumerate(texts)])
    _bootstrap.emit("Paraphrase traffic / PAWS (40K)",
                    {"prefix": 27.67, "pic": 0.00, "pic_proc": 0.00,
                     "in_sess": 0.00, "cross_sess": 0.00}, d)


if __name__ == "__main__":
    main()
