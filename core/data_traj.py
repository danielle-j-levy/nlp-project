"""Coding-agent trajectory loaders: file-read observations from three corpora.

Corpus A  SWE-bench/SWE-smith-trajectories  -- SWE-bench-family task instances,
          SWE-agent-style scaffold, sandbox root /testbed.
Corpus B  nebius/SWE-rebench-openhands-trajectories -- SWE-rebench task
          instances, OpenHands scaffold, sandbox root /workspace/<instance>.
Corpus C  zai-org/CC-Bench-trajectories -- Claude Code scaffold on CC-Bench,
          five models x 74 tasks. Not repository bug-fixing at all: greenfield
          web/app/ML development, where most files the agent reads are ones it
          wrote itself minutes earlier.

A and B render file contents through a `cat -n` viewer; C uses Claude Code's
`   12<U+2192>text` form and pairs results to calls by tool-use id. Three
scaffolds, three observation formats, two different kinds of task -- a boundary
rule that wins across all of them is not fitting one harness's formatting.

Every loader yields the same shape, so the analysis code never learns which
corpus it is reading:

    {traj_id, instance_id, repo, reads: [(path, [(lineno, text), ...]), ...],
     msg_chars, read_chars, n_msgs}

`repo` is the grouping key for "another episode working on the same thing":
the repository for A and B, the task id for C (whose five model runs per task
are the analogue of several episodes over one codebase).
"""

import json
import re

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from data import load_hf_token

# A/B: `cat -n` viewer, path in a header line, content as `<n>\t<text>`
HEAD = re.compile(r"result of running `cat -n` on (?:a snippet of )?([^\n:]+):")
LINE = re.compile(r"^\s*(\d+)\t(.*)$")
# C: Claude Code viewer, no header (path comes from the tool call), `<n>→<text>`
ARROW = re.compile(r"^\s*(\d+)→(.*)$")
# D: classic SWE-agent editor, a 100-line window with its own header and `<n>:<text>`
SWEA_HEAD = re.compile(r"\[File: ([^\n\]]+?) \(\d+ lines total\)\]")
SWEA_LINE = re.compile(r"^\s*(\d+):(.*)$")

SANDBOX = re.compile(r"^(?:/testbed/|/workspace/[^/]+/|/workspace/|/app/projects/|/app/)")

CORPORA = {
    "swesmith": {
        "repo": "SWE-bench/SWE-smith-trajectories",
        "files": [f"data/train-{i:05d}-of-00008.parquet" for i in range(8)],
        "cols": ["messages", "instance_id"],
        "format": "cat_n",
    },
    "openhands": {
        "repo": "nebius/SWE-rebench-openhands-trajectories",
        "files": ["trajectories.parquet"],
        "cols": ["trajectory", "instance_id"],
        "format": "cat_n",
    },
    "sweagent": {
        "repo": "nebius/SWE-agent-trajectories",
        "files": [f"data/train-{i:05d}-of-00012.parquet" for i in range(12)],
        "cols": ["trajectory", "instance_id"],
        "format": "swe_agent",
    },
    "ccbench": {
        "repo": "zai-org/CC-Bench-trajectories",
        "files": ["train.parquet"],
        "cols": ["trajectory", "task_id"],
        "format": "claude_code",
    },
}


def norm_path(p):
    """Drop the sandbox root. Only has to be self-consistent within an episode:
    file state is tracked per episode, and span keys are content hashes."""
    return SANDBOX.sub("", (p or "").strip())


def repo_of(instance_id):
    """`<org>__<repo>-<n>` (SWE-bench / SWE-rebench) or `<org>__<repo>.<commit>.<mutation>`
    (SWE-smith) -> `<org>/<repo>`. Used to tell same-repo cross-trajectory reuse
    from reuse across unrelated projects."""
    head = str(instance_id).split("__", 1)
    if len(head) < 2:
        return str(instance_id)
    base = head[1].split(".", 1)[0]        # drop SWE-smith commit/mutation suffix
    base = re.sub(r"-\d+$", "", base)      # drop SWE-bench instance number
    return f"{head[0]}/{base}"


def _numbered(block, pat):
    """Leading run of numbered lines in `block`. Stops at the first line that is
    not numbered, which is what ends a viewer block -- a footer, a clip notice,
    Claude Code's trailing <system-reminder>, or ordinary prose."""
    out = []
    for raw in block.split("\n"):
        m = pat.match(raw)
        if m:
            out.append((int(m.group(1)), m.group(2)))
        elif out:
            break
    return out


def parse_reads(content):
    """All `cat -n` blocks in one observation, as (path, [(lineno, text), ...]).

    The viewer's line-number prefix is stripped here: it is a rendering artifact,
    and keeping it would make every span below an inserted line unreusable for
    reasons that have nothing to do with the code. The numbers are retained
    separately as the absolute file coordinate the anchored methods need.
    """
    if not content or "cat -n" not in content:
        return []
    heads = list(HEAD.finditer(content))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(content)
        lines = _numbered(content[m.end():end], LINE)
        if len(lines) >= 2:
            out.append((norm_path(m.group(1)), lines))
    return out


def _text_of(content):
    """Claude Code content blocks -> plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in content)
    return ""


def _walk_cat_n(msgs):
    reads, chars = [], 0
    for m in msgs or []:
        c = m.get("content") or ""
        chars += len(c)
        reads.extend(parse_reads(c))
    return reads, chars, len(msgs or [])


def _walk_swe_agent(msgs):
    """Classic SWE-agent editor output: a `[File: ...]` header, an optional
    "(N more lines above)" notice, then `<lineno>:<text>` for a 100-line window."""
    reads, chars = [], 0
    for m in msgs or []:
        s = m.get("text") or ""
        chars += len(s)
        if "[File: " not in s:
            continue
        heads = list(SWEA_HEAD.finditer(s))
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(s)
            lines = _numbered(s[h.end():end], SWEA_LINE)
            if len(lines) >= 2:
                reads.append((norm_path(h.group(1)), lines))
    return reads, chars, len(msgs or [])


def _walk_claude_code(events):
    """Claude Code JSONL events. A viewer result carries no path of its own, so
    each tool_result is matched back to the tool_use that produced it."""
    uses, reads, chars = {}, [], 0
    for ev in events or []:
        msg = ev.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                chars += len(b.get("text") or "")
            elif b.get("type") == "tool_use":
                uses[b.get("id")] = (b.get("input") or {}).get("file_path")
            elif b.get("type") == "tool_result":
                s = _text_of(b.get("content"))
                chars += len(s)
                lines = _numbered(s, ARROW)
                if len(lines) >= 2:
                    reads.append((norm_path(uses.get(b.get("tool_use_id")) or "?"), lines))
    return reads, chars, len(events or [])


def classify_tool(text):
    """Which tool produced this observation, from its text alone.

    Only the Claude Code corpus records tool names structurally; the other three
    flatten every observation to plain text. These signatures are the scaffolds'
    own fixed templates, so they identify the producing tool exactly wherever
    they match, and fall through to "other" where they do not."""
    s = text[:400].lstrip()
    if s.startswith("OBSERVATION:"):                 # swesmith / openhands wrapper
        s = s[len("OBSERVATION:"):].lstrip()
    if s.startswith("[File:") or "(Open file:" in text[:80]:
        return "file_read"                           # SWE-agent editor window
    if "result of running `cat -n`" in s[:200]:
        # An edit echoes the patched file back through the same viewer, so the
        # edit banner must be tested first or every edit counts as a read.
        return "edit" if "has been edited" in s[:200] else "file_read"
    if s.startswith(("File created successfully", "The file ", "Your proposed edit",
                     "ERROR: No changes were made", "No replacement was performed")):
        return "edit"
    if s.startswith(("Here's the files and directories", "Found ", "No matches found",
                     "grep:")):
        return "search_list"
    if s.startswith(("Your thought has been logged", "<uploaded_files>", "ISSUE:",
                     "We're currently solving")):
        return "prompt_task"
    return "bash_other"


def iter_messages(corpus, max_trajs=None, one_per_task=False, with_tools=False):
    """Yield every message of each trajectory as (role, text), role in
    {"system", "input", "decode"}.

    "decode" is text the model generated: its KV is built one token at a time
    during decoding and is never prefilled, so no cache can eliminate it. It is
    separated here so it can be kept out of both numerator and denominator when
    the metric is a fraction of the prefill workload.

    `with_tools` yields (role, text, tool, call) instead, where tool names the
    source of the message: the recorded tool name on Claude Code, a signature
    match on the other scaffolds, "system" / "user" for non-tool input, and "" for
    decode. `call` is the arguments of the tool call that produced this
    observation, matched by tool_use_id where the scaffold records one. It is
    empty on the scaffolds that inline the call into the assistant's text, where
    the preceding decode message already contains it."""
    cfg = CORPORA[corpus]
    load_hf_token()
    msg_col, id_col = cfg["cols"]
    fmt = cfg["format"]
    seen_ids = set()
    n = 0
    for rf in cfg["files"]:
        fs = HfFileSystem(skip_instance_cache=True)
        with fs.open(f"datasets/{cfg['repo']}/{rf}") as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=cfg["cols"])
                for raw, iid in zip(t.column(msg_col).to_pylist(),
                                    t.column(id_col).to_pylist()):
                    if one_per_task:
                        if str(iid) in seen_ids:
                            continue
                        seen_ids.add(str(iid))
                    msgs = json.loads(raw) if isinstance(raw, str) else raw
                    out = []
                    if fmt == "claude_code":
                        names, calls = {}, {}
                        for ev in msgs or []:
                            m = ev.get("message") or {}
                            role = m.get("role")
                            c = m.get("content")
                            if isinstance(c, str):
                                out.append(("decode" if role == "assistant" else "input", c,
                                            "" if role == "assistant" else "user", ""))
                            elif isinstance(c, list):
                                for b in c:
                                    if not isinstance(b, dict):
                                        continue
                                    if b.get("type") == "text":
                                        out.append(("decode" if role == "assistant" else "input",
                                                    b.get("text") or "",
                                                    "" if role == "assistant" else "user", ""))
                                    elif b.get("type") == "tool_result":
                                        # match the call by id: one assistant message
                                        # may carry several tool_use blocks, so the
                                        # nearest preceding one is not necessarily this
                                        # result's own call
                                        tid = b.get("tool_use_id")
                                        out.append(("input", _text_of(b.get("content")),
                                                    names.get(tid, "unknown"),
                                                    calls.get(tid, "")))
                                    elif b.get("type") == "tool_use":
                                        names[b.get("id")] = b.get("name") or "unknown"
                                        calls[b.get("id")] = json.dumps(b.get("input") or {})
                                        out.append(("decode",
                                                    json.dumps(b.get("input") or {}), "", ""))
                    elif fmt == "swe_agent":
                        for m in msgs or []:
                            if m.get("system_prompt"):
                                out.append(("system", m["system_prompt"], "system", ""))
                            if m.get("text"):
                                ai = m.get("role") == "ai"
                                out.append(("decode" if ai else "input", m["text"],
                                            "" if ai else classify_tool(m["text"]), ""))
                    else:
                        for m in msgs or []:
                            c = m.get("content") or ""
                            r = m.get("role")
                            out.append(("system" if r == "system"
                                        else "decode" if r == "assistant" else "input", c,
                                        "system" if r == "system"
                                        else "" if r == "assistant" else classify_tool(c), ""))
                    out = [(r, x, t, k) for r, x, t, k in out if x]
                    if not with_tools:
                        out = [(r, x) for r, x, _, _ in out]
                    if not out:
                        continue
                    yield {"traj_id": n, "instance_id": str(iid),
                           "repo": (f"task-{iid}" if fmt == "claude_code"
                                    else repo_of(iid or "")),
                           "messages": out}
                    n += 1
                    if max_trajs and n >= max_trajs:
                        return


def iter_trajectories(corpus, max_trajs=None, one_per_task=False):
    """Yield one dict per trajectory that contains at least one file read.

    `one_per_task` keeps only the first episode per instance id. Several corpora
    run the same task more than once -- CC-Bench evaluates five models on each of
    its 74 tasks, OpenHands repeats 77 instances -- and those repeats read the
    same starter files, so leaving them in credits the cache with reuse that is
    an artefact of how the benchmark was assembled rather than a property of the
    workload. Note this is not the same as collapsing episodes that merely share
    a repository: different tasks on one codebase are exactly the realistic
    cross-session win, and stay."""
    cfg = CORPORA[corpus]
    load_hf_token()
    msg_col, id_col = cfg["cols"]
    walk = {"claude_code": _walk_claude_code,
            "swe_agent": _walk_swe_agent}.get(cfg["format"], _walk_cat_n)
    seen_ids = set()
    n = 0
    for rf in cfg["files"]:
        # A fresh filesystem per shard: fsspec caches HfFileSystem instances, and
        # a cached one's HTTP client can be closed under us between shards, which
        # surfaces as "Cannot send a request, as the client has been closed"
        # partway through a multi-shard corpus.
        fs = HfFileSystem(skip_instance_cache=True)
        with fs.open(f"datasets/{cfg['repo']}/{rf}") as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=cfg["cols"])
                for raw, iid in zip(t.column(msg_col).to_pylist(),
                                    t.column(id_col).to_pylist()):
                    if one_per_task:
                        if str(iid) in seen_ids:
                            continue
                        seen_ids.add(str(iid))
                    msgs = json.loads(raw) if isinstance(raw, str) else raw
                    reads, chars, nmsg = walk(msgs)
                    if not reads:
                        continue
                    yield {
                        "traj_id": n,
                        "instance_id": str(iid),
                        # CC-Bench runs five models over each task, so the task
                        # id plays the part the repository plays in A and B
                        "repo": (f"task-{iid}" if cfg["format"] == "claude_code"
                                 else repo_of(iid or "")),
                        "reads": reads,
                        "msg_chars": chars,
                        "read_chars": sum(len(t2) + 1 for _, ls in reads for _, t2 in ls),
                        "n_msgs": nmsg,
                    }
                    n += 1
                    if max_trajs and n >= max_trajs:
                        return
