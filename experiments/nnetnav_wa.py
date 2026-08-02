"""Row: Web pages (a11y tree) / NNetNav-WA (WebArena).

Per-step prompts share a fixed instruction head, so reuse is measured on the
coding-row denominator (head counted once per session, each observation once;
16-token prefix chain for prefix-cache credit, >=500-token line spans for PIC on
the remainder). The `replace` treatment rewrites volatile element handles ([123])
into stable content-derived handles. PIC-proc / in-sess / cross-sess are the
treated numbers; PIC is the untreated pic.

Published: Prefix 37.32%  PIC 4.50%  PIC-proc 10.03%  in-sess 5.62  cross-sess 4.41
Data: HF stanfordnlp/nnetnav-wa (train.jsonl). Requires HF_TOKEN.
"""
import _bootstrap
from web_sessionhead import run, replace_handles

REPO = "stanfordnlp/nnetnav-wa"


def main():
    v = run(REPO, None)
    t = run(REPO, replace_handles)
    _bootstrap.emit(
        "Web pages (a11y tree) / NNetNav-WA (WebArena)",
        {"prefix": 37.32, "pic": 4.50, "pic_proc": 10.03,
         "in_sess": 5.62, "cross_sess": 4.41},
        {"prefix_caching_pct": t["prefix_caching_pct"], "pic_pct": v["pic_pct"],
         "pic_proc_pct": t["pic_pct"], "in_session": t["pic_in_sess_pct"],
         "cross_session": t["pic_cross_sess_pct"]})


if __name__ == "__main__":
    main()
