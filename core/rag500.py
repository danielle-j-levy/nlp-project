import argparse, glob, json, os
from data import load_hf_token; load_hf_token()
from pic_rules import lines_of, report, measure, nt


def sessions_multidoc2dial():
    docs = json.load(open("data/multidoc2dial/multidoc2dial_doc.json"))["doc_data"]
    doc_text = {}
    for domain, dd in docs.items():
        for doc_id, d in dd.items():
            doc_text[doc_id] = d["doc_text"]
    out = []
    for split in ("train", "validation"):
        data = json.load(open(f"data/multidoc2dial/multidoc2dial_dial_{split}.json"))["dial_data"]
        for domain, dials in data.items():
            for dial in dials:
                snaps = []
                for t in dial["turns"]:
                    if t.get("role") != "user":
                        continue
                    seen = set()
                    for r in (t.get("references") or []):
                        did = r.get("doc_id")
                        if did and did not in seen and did in doc_text:
                            seen.add(did)
                            snaps.append(lines_of(doc_text[did]))
                if len(snaps) >= 1:
                    out.append({"task": dial["dial_id"], "snaps": snaps})
    return out


def sessions_mtrag(which="human"):
    convs = json.load(open(f"data/mtrag/conversations/conversations_{which}.json"))
    out = []
    for i, c in enumerate(convs):
        snaps = []
        for m in c.get("messages") or []:
            ctxs = m.get("contexts") or []
            if not ctxs:
                continue
            units = [p.get("text") or "" for p in ctxs if (p.get("text") or "").strip()]
            if units:
                snaps.append(units)
        if snaps:
            out.append({"task": f"{c.get('domain','')}/{i}", "snaps": snaps})
    return out


def tau_docs():
    d = {}
    for f in glob.glob("data/tauknowledge/banking_knowledge/documents/*.json"):
        j = json.load(open(f))
        d[j["id"]] = j.get("title", "") + "\n" + (j.get("content") or "")
    if not d:
        p = "data/tauknowledge/banking_knowledge/documents"
        for f in glob.glob(os.path.join(p, "*")):
            try:
                j = json.load(open(f))
                d[j["id"]] = j.get("title", "") + "\n" + (j.get("content") or "")
            except Exception:
                pass
    return d


def sessions_tau_rag():
    docs = tau_docs()
    tasks = json.load(open("data/tauknowledge/banking_knowledge/tasks.json"))
    out = []
    for t in tasks:
        req = t.get("required_documents") or []
        units = [docs[d] for d in sorted(req) if d in docs]
        if units:
            out.append({"task": t["id"], "snaps": [sum([lines_of(u) for u in units], [])]})
    return out, docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["multidoc2dial", "mtrag", "tau"])
    args = ap.parse_args()
    R = {}
    if args.part == "multidoc2dial":
        sessions = sessions_multidoc2dial()
        print(f"multidoc2dial: {len(sessions)} dialogues, "
              f"{sum(len(s['snaps']) for s in sessions)} grounded turns", flush=True)
        R = report("multidoc2dial", sessions)
        for k in ("all_runs", "one_run_per_task"):
            print(f"  {k}: {json.dumps(R[k])}", flush=True)
        print(f"  duplicate_runs: {R['duplicate_runs']}", flush=True)
    elif args.part == "mtrag":
        for which in ("human", "synthetic"):
            sessions = sessions_mtrag(which)
            print(f"mtrag_{which}: {len(sessions)} conversations, "
                  f"{sum(len(s['snaps']) for s in sessions)} retrieval turns", flush=True)
            R[which] = report("mtrag_" + which, sessions)
            for k in ("all_runs", "one_run_per_task"):
                print(f"  {k}: {json.dumps(R[which][k])}", flush=True)
    else:
        sessions, docs = sessions_tau_rag()
        sizes = [nt(v) for v in docs.values()]
        print(f"tau: {len(sessions)} tasks, {len(docs)} policy docs, "
              f"doc tokens min/med/max = {min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}",
              flush=True)
        R["rag_per_task_docs"] = report("tau_rag", sessions)
        for k in ("all_runs", "one_run_per_task"):
            print(f"  rag {k}: {json.dumps(R['rag_per_task_docs'][k])}", flush=True)
        kb_units = [docs[k] for k in sorted(docs)]
        kb_sessions = [{"task": s["task"],
                        "snaps": [sum([lines_of(u) for u in kb_units], [])]}
                       for s in sessions]
        R["full_kb_prompt"] = measure(kb_sessions)
        print(f"  full-KB-in-prompt: {json.dumps(R['full_kb_prompt'])}", flush=True)
    json.dump(R, open(f"report/rag_{args.part}.json", "w"), indent=2)
    print("DONERAG_" + args.part)


if __name__ == "__main__":
    main()
