"""Reuse measured under the five accounting rules, with no departures.

  1. Second occurrence onward. The first emission of a span populates the cache
     and scores zero; only later occurrences count.
  2. Divide by all tokens. The denominator is every token of every trajectory
     evaluated, each counted once -- system prompt, user turns, tool
     observations and assistant turns alike -- not only the tokens of the
     content stream being partitioned.
  3. De-duplication. One trajectory per task.
  4. Spans of >= 500 tokens. Shorter spans are delivered but never cached, so
     they can never be credited.
  5. Exclude reuse already served by same-session prefix caching. A multi-turn
     prompt is the previous prompt plus new text, so everything before the new
     text is already resident and is never recomputed. Each turn therefore
     contributes only its newly appended text, both to the numerator and to the
     denominator, and a span counts as reused only when it recurs at a shifted
     position or in a different session.

Rules 2 and 5 together mean each token enters the denominator exactly once. A
naive reading of rule 2 -- summing prompt lengths across turns -- would count a
20-turn conversation's prefix twenty times and divide every result by roughly
ten.

Spans are >= 500-token line-aligned packs taken from the start of each newly
appended message, with a trailing remainder merged back so no span falls under
the floor. That is the untreated baseline: no bucketing, no alignment.
"""

import argparse
import collections
import json
import os
import time

from transformers import AutoTokenizer

from data import load_hf_token
from data_traj import iter_messages
from spans import h64, m_pack_tokens

MIN_SPAN = 500


def run(corpus, tokenizer_id, max_trajs, one_per_task=True, log_every=200,
        system_as_prefix=False):
    load_hf_token()
    tok = AutoTokenizer.from_pretrained(tokenizer_id, token=os.environ.get("HF_TOKEN"))
    t0 = time.time()
    tcache = {}

    def ntok(s):
        v = tcache.get(s)
        if v is None:
            v = len(tok(s + "\n", add_special_tokens=False)["input_ids"])
            if len(tcache) < 4_000_000:
                tcache[s] = v
        return v

    seen = {}                       # span hash -> (traj id, repo id)
    repos = {}
    total = reused = within = cross_same = cross_diff = 0
    uncacheable = nspans = sys_prefix = 0
    ntraj = nmsg = 0

    for tr in iter_messages(corpus, max_trajs, one_per_task=one_per_task):
        ntraj += 1
        rid = repos.setdefault(tr["repo"], len(repos))
        for role, text in tr["messages"]:
            # rule 2: the denominator is prefilled input tokens. Model-generated
            # text has its KV built during decode and is never prefilled, so it
            # belongs to neither numerator nor denominator.
            if role == "decode":
                continue
            # A system prompt is the leading portion of every request in the
            # session AND of every other session using the same scaffold, so a
            # shared prefix cache already serves it at position 0. Rule 5 says
            # that reuse is not PIC's to claim. Its tokens still enter the
            # denominator once (rule 2); they just never score.
            if role == "system" and system_as_prefix:
                sys_tokens = sum(ntok(ln) for ln in text.split("\n"))
                total += sys_tokens
                sys_prefix += sys_tokens
                continue
            nmsg += 1
            new = {}
            lines = text.split("\n")
            cum = [0]
            for ln in lines:
                cum.append(cum[-1] + ntok(ln))
            if not cum[-1]:
                continue
            # rule 5: this message is the turn's newly appended text, so it is
            # scored once, here, and never again as part of a later prompt
            for a, b in m_pack_tokens(cum, MIN_SPAN, min_tail=MIN_SPAN):
                nt = cum[b] - cum[a]
                total += nt
                nspans += 1
                if nt < MIN_SPAN:          # rule 4: too short to be cacheable
                    uncacheable += nt
                    continue
                key = h64("\n".join(lines[a:b]))
                prev = seen.get(key)
                if prev is None:
                    new.setdefault(key, nt)
                else:                       # rule 1: second occurrence onward
                    reused += nt
                    if prev[0] == tr["traj_id"]:
                        within += nt
                    elif prev[1] == rid:
                        cross_same += nt
                    else:
                        cross_diff += nt
            # Insert at the END OF THE MESSAGE, not the end of the trajectory: a
            # later turn of the same session must be able to hit content this
            # turn introduced (rule 5 counts text reappearing at a shifted
            # position), while a repeat inside one message is not reuse -- the
            # engine prefills that message once either way.
            for k in new:
                seen.setdefault(k, (tr["traj_id"], rid))
            new = {}
        if log_every and ntraj % log_every == 0:
            print(f"  {ntraj} trajectories, {total:,} tokens, {time.time()-t0:.0f}s",
                  flush=True)

    p = lambda a: round(100.0 * a / total, 2) if total else 0.0
    return {
        "corpus": corpus,
        "trajectories": ntraj,
        "prefilled_messages": nmsg,
        "distinct_groups": len(repos),
        "total_tokens": total,
        "reused_tokens": reused,
        "reuse_pct": p(reused),
        "in_session_pct": p(within),
        "cross_session_same_group_pct": p(cross_same),
        "cross_session_other_group_pct": p(cross_diff),
        "cross_session_pct": p(cross_same + cross_diff),
        "uncacheable_pct": p(uncacheable),
        "system_prefix_tokens": sys_prefix,
        "system_prefix_pct": p(sys_prefix),
        "system_as_prefix": system_as_prefix,
        "mean_span_tokens": round(total / nspans, 1) if nspans else 0,
        "unique_spans": len(seen),
        "min_span_tokens": MIN_SPAN,
        "one_per_task": one_per_task,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    choices=["swesmith", "openhands", "ccbench", "sweagent"])
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-trajs", type=int, default=1000)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--system-as-prefix", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = run(args.corpus, args.tokenizer, args.max_trajs, not args.no_dedup,
            system_as_prefix=args.system_as_prefix)
    print(json.dumps(r, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(r, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
