"""Row: Web pages (live websites) / NNetNav-Live.

Accessibility-tree observations from live sites, measured on the same head-once
denominator as the WebArena row (16-token prefix chain, >=500-token line spans on
the remainder). Same `replace` treatment (volatile [123] handles -> stable
content handles). PIC-proc / in-sess / cross-sess are the treated numbers.

Published: Prefix 20.31%  PIC 11.82%  PIC-proc 14.79%  in-sess 11.79  cross-sess 3.00
Data: HF stanfordnlp/nnetnav-live (train.jsonl). Requires HF_TOKEN.
"""
import _bootstrap
from web_sessionhead import run, replace_handles

REPO = "stanfordnlp/nnetnav-live"


def main():
    v = run(REPO, None)
    t = run(REPO, replace_handles)
    _bootstrap.emit(
        "Web pages (live websites) / NNetNav-Live",
        {"prefix": 20.31, "pic": 11.82, "pic_proc": 14.79,
         "in_sess": 11.79, "cross_sess": 3.00},
        {"prefix_caching_pct": t["prefix_caching_pct"], "pic_pct": v["pic_pct"],
         "pic_proc_pct": t["pic_pct"], "in_session": t["pic_in_sess_pct"],
         "cross_session": t["pic_cross_sess_pct"]})


if __name__ == "__main__":
    main()
